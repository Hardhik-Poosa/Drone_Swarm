"""
pipeline_video.py — Drone Swarm Video Formation Generator
==========================================================
Takes a video (MP4/AVI/…) and produces an output video where a swarm of drones
traces the person/object silhouette in every frame, smoothly following motion.

Pipeline per frame
------------------
1. Resize frame to processing resolution
2. DeepLabV3 → person mask
3. Poisson-disk → N target 2D positions inside mask
4. MiDaS depth → Z coordinate per target  (every DEPTH_EVERY frames)
5. Hungarian assignment: current drone positions → closest set of targets
6. Physics/RL motion: run MOTION_STEPS steps toward targets
7. Render: side-by-side (original | drone overlay) + optional 3D axes panel
8. Write rendered frame to output video

Usage
-----
    python pipeline_video.py --video input_images/4761762-uhd_2160_4096_25fps.mp4
    python pipeline_video.py --video input_images/4761762-uhd_2160_4096_25fps.mp4 \\
        --drones 400 --rl-checkpoint RL/checkpoints/sb3/best_model \\
        --render-3d --output output/
"""

import os, sys, argparse, time, math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── lazy heavy imports (loaded once, reused across all frames) ───────────────
_seg_model   = None
_seg_tfm     = None
_midas_model = None
_midas_tfm   = None
_rl_ctrl     = None

PERSON_CLASS = 15

# ────────────────────────────────────────────────────────────────────────────
# Model loaders (lazy)
# ────────────────────────────────────────────────────────────────────────────

def _load_segmentation():
    global _seg_model, _seg_tfm
    if _seg_model is not None:
        return
    print("[load] DeepLabV3-ResNet101…")
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    _seg_model = tvm.segmentation.deeplabv3_resnet101(weights=weights).eval()
    _seg_tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    print("[load] Segmentation model ready.")


def _load_midas():
    global _midas_model, _midas_tfm
    if _midas_model is not None:
        return
    print("[load] MiDaS DPT-Large…")
    import torch
    _midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True).eval()
    transforms   = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    _midas_tfm   = transforms.dpt_transform
    print("[load] Depth model ready.")


def _load_rl(ckpt_path: str):
    global _rl_ctrl
    if _rl_ctrl is not None:
        return
    print(f"[load] RL controller from {ckpt_path}…")
    from RL.rl_controller import RLController
    _rl_ctrl = RLController(ckpt_path)
    print("[load] RL controller ready.")


# ────────────────────────────────────────────────────────────────────────────
# Segmentation — returns (H, W) bool mask for person class
# ────────────────────────────────────────────────────────────────────────────

def segment_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """Returns bool mask (H x W) at the same resolution as bgr_frame."""
    import torch
    from PIL import Image as PILImage

    _load_segmentation()
    h, w = bgr_frame.shape[:2]
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb).resize((512, 512))
    inp = _seg_tfm(np.array(pil)).unsqueeze(0)
    with torch.no_grad():
        out = _seg_model(inp)["out"][0].argmax(0).numpy()
    # resize mask back to original frame size
    mask512 = (out == PERSON_CLASS).astype(np.uint8)
    mask = cv2.resize(mask512, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    # morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  k, iterations=1)
    return mask_u8.astype(bool)


# ────────────────────────────────────────────────────────────────────────────
# Depth estimation — returns float32 depth map in [0, 1]
# ────────────────────────────────────────────────────────────────────────────

def depth_frame(bgr_frame: np.ndarray) -> np.ndarray:
    import torch

    _load_midas()
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    inp = _midas_tfm(rgb)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        pred = _midas_model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    d -= d.min()
    if d.max() > 0:
        d /= d.max()
    d = cv2.GaussianBlur(d, (9, 9), 0)
    return d


# ────────────────────────────────────────────────────────────────────────────
# Poisson-disk drone target sampling inside a mask
# ────────────────────────────────────────────────────────────────────────────

def _poisson_disk(mask: np.ndarray, n: int, min_dist_ratio: float = 0.4) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask — no person detected in frame.")
    area = len(xs)
    cell = max(1.0, math.sqrt(area / max(n, 1)) * min_dist_ratio)
    min_sq = cell ** 2
    rng = np.random.default_rng(0)
    order = rng.permutation(len(xs))
    xs_, ys_ = xs[order], ys[order]
    accepted, grid = [], {}

    def _key(x, y):
        return (int(x // cell), int(y // cell))

    for xi, yi in zip(xs_, ys_):
        cx, cy = _key(xi, yi)
        neighbor_keys = [(cx+dx, cy+dy) for dx in range(-2, 3) for dy in range(-2, 3)]
        ok = True
        for nk in neighbor_keys:
            for nx_, ny_ in grid.get(nk, []):
                if (xi - nx_) ** 2 + (yi - ny_) ** 2 < min_sq:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            accepted.append((xi, yi))
            grid.setdefault(_key(xi, yi), []).append((xi, yi))
            if len(accepted) >= n:
                break

    pts = np.array(accepted, dtype=np.float32)
    # if we got fewer than n, fill remainder from mask pixels
    if len(pts) < n:
        need = n - len(pts)
        idx  = rng.choice(len(xs), size=need, replace=True)
        extra = np.column_stack([xs[idx], ys[idx]]).astype(np.float32)
        pts = np.vstack([pts, extra])
    return pts  # (n, 2)  as (col=x, row=y)


def _outline_pts(mask: np.ndarray, n: int) -> np.ndarray:
    cnts, _ = cv2.findContours(mask.astype(np.uint8)*255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2), np.float32)
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    arclen = np.cumsum(np.r_[0, np.linalg.norm(np.diff(cnt, axis=0), axis=1)])
    total  = arclen[-1]
    if total < 1e-6 or n == 0:
        return np.zeros((0, 2), np.float32)
    ts  = np.linspace(0, total, n, endpoint=False)
    xi  = np.interp(ts, arclen, cnt[:, 0])
    yi  = np.interp(ts, arclen, cnt[:, 1])
    return np.column_stack([xi, yi]).astype(np.float32)


def sample_targets(mask: np.ndarray, n_drones: int) -> np.ndarray:
    """Return (n_drones, 2) target positions (pixel x, y) inside/on the mask."""
    n_out = max(1, n_drones // 4)
    n_in  = n_drones - n_out
    pts_in  = _poisson_disk(mask, n_in)
    pts_out = _outline_pts(mask, n_out)
    pts = np.vstack([pts_in, pts_out]) if len(pts_out) else pts_in
    # normalize to a unit-ish coordinate system for the physics/RL engine
    hw, hh = mask.shape[1] / 2.0, mask.shape[0] / 2.0
    pts_norm = (pts - np.array([hw, hh])) / max(hw, hh) * 5.0
    if len(pts_norm) < n_drones:
        pts_norm = np.vstack([pts_norm,
                              pts_norm[:(n_drones - len(pts_norm))]])
    return pts_norm[:n_drones].astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Physics motion step (vectorised, O(N log N))
# ────────────────────────────────────────────────────────────────────────────

def physics_step(positions: np.ndarray,
                 targets:   np.ndarray,
                 step_size: float = 0.12,
                 rep_strength: float = 0.04,
                 rep_radius:   float = 0.6) -> np.ndarray:
    attraction = targets - positions
    repulsion  = np.zeros_like(positions)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=rep_radius, output_type="ndarray")
    if len(pairs):
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        diff  = positions[i_idx] - positions[j_idx]
        dist  = np.linalg.norm(diff, axis=1, keepdims=True).clip(min=0.1)  # clamp min dist
        force = diff / (dist ** 2) * rep_strength   # 1/r^2 (not 1/r^3) — gentler
        force = np.clip(force, -0.5, 0.5)           # hard cap per pair
        np.add.at(repulsion, i_idx,  force)
        np.add.at(repulsion, j_idx, -force)
    return (positions + step_size * attraction + repulsion).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Hungarian reassignment (keeps temporal consistency)
# ────────────────────────────────────────────────────────────────────────────

def hungarian_reassign(positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return targets reordered so drone i's target minimises total travel."""
    cost = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=2)
    _, col = linear_sum_assignment(cost)
    return targets[col]


# ────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ────────────────────────────────────────────────────────────────────────────

def _norm_to_pixel(pts_norm: np.ndarray, frame_h: int, frame_w: int):
    """Convert normalised coords back to pixel coords on the frame."""
    hw, hh = frame_w / 2.0, frame_h / 2.0
    sc = max(hw, hh) / 5.0
    px = (pts_norm[:, 0] * sc + hw).astype(int)
    py = (pts_norm[:, 1] * sc + hh).astype(int)
    return np.clip(px, 0, frame_w - 1), np.clip(py, 0, frame_h - 1)


def _depth_at_pts(depth_map: np.ndarray, px, py) -> np.ndarray:
    """Sample depth map at pixel positions. Returns float32 array [0..1]."""
    return depth_map[py, px].astype(np.float32)


def _drone_color(depth_val: float) -> tuple:
    """Map depth [0..1] to a BGR colour (hot colormap: close=bright, far=dark)."""
    v = float(np.clip(1.0 - depth_val, 0.0, 1.0))   # invert: near=bright
    r = int(np.clip(v * 255, 0, 255))
    g = int(np.clip((v - 0.33) * 3 * 255, 0, 255))
    b = int(np.clip((v - 0.66) * 3 * 255, 0, 255))
    return (b, g, r)   # BGR


def render_overlay(bgr_frame: np.ndarray,
                   pos_norm: np.ndarray,
                   depth_map: np.ndarray | None,
                   dot_radius: int = 4,
                   alpha: float = 0.55) -> np.ndarray:
    """Draw drone positions as coloured dots onto the frame."""
    h, w = bgr_frame.shape[:2]
    overlay = bgr_frame.copy()
    px, py = _norm_to_pixel(pos_norm, h, w)

    for i, (x, y) in enumerate(zip(px.tolist(), py.tolist())):
        if depth_map is not None:
            dv = float(depth_map[y, x])
            col = _drone_color(dv)
        else:
            col = (0, 220, 50)   # default: bright green
        cv2.circle(overlay, (x, y), dot_radius, col, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), dot_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    out = cv2.addWeighted(bgr_frame, 1 - alpha, overlay, alpha, 0)
    return out


def render_3d_axes(pos_norm: np.ndarray,
                   depth_vals: np.ndarray | None,
                   canvas_hw: tuple) -> np.ndarray:
    """
    Draw a lightweight 3D scatter using a simple isometric projection.
    Returns a BGR numpy image of size canvas_hw.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    H, W = canvas_hw
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")

    xs = pos_norm[:, 0]
    ys = pos_norm[:, 1]
    zs = depth_vals if depth_vals is not None else np.zeros(len(xs))
    zs = zs * 3.0   # scale depth for visual emphasis

    sc = ax.scatter(xs, ys, zs, c=zs, cmap="plasma",
                    s=12, alpha=0.85, edgecolors="none")
    ax.set_xlabel("X", color="white", fontsize=6)
    ax.set_ylabel("Y", color="white", fontsize=6)
    ax.set_zlabel("Z", color="white", fontsize=6)
    ax.tick_params(colors="white", labelsize=5)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor("#333333")
    ax.view_init(elev=25, azim=-60)
    ax.set_title("Swarm 3D", color="white", fontsize=8, pad=2)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 4)
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def make_hud(frame_idx: int, n_drones: int,
             mean_err: float, fps: float,
             rl_active: bool, h: int, w: int) -> np.ndarray:
    """Transparent HUD bar at the top of a frame."""
    bar = np.zeros((50, w, 3), dtype=np.uint8)
    mode = "RL-PPO" if rl_active else "Physics"
    txt  = (f"  Frame {frame_idx:04d} | Drones {n_drones} | "
            f"Err {mean_err:.3f} | {mode} | {fps:.1f} fps")
    cv2.putText(bar, txt, (5, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 180), 1, cv2.LINE_AA)
    return bar


# ────────────────────────────────────────────────────────────────────────────
# Main video pipeline
# ────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    output_dir: str      = "output",
    n_drones: int        = 300,
    proc_width: int      = 640,          # downscale for model inference
    motion_steps: int    = 6,            # physics/RL steps per video frame
    depth_every: int     = 5,            # run MiDaS every N frames
    rl_checkpoint: str | None = None,
    render_3d: bool      = False,
    max_frames: int      = 0,            # 0 = all frames
    dot_radius: int      = 5,
    step_size: float     = 0.15,
    rep_strength: float  = 0.05,
    rep_radius: float    = 0.5,
    skip_frames: int     = 0,            # skip every N frames for speed (0=none)
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    video_stem = Path(video_path).stem
    out_path   = str(Path(output_dir) / f"{video_stem}_swarm.mp4")
    csv_path   = str(Path(output_dir) / f"{video_stem}_positions.csv")

    # ── open input video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS)
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_fr = min(total_fr, max_frames)
    out_fps = src_fps / max(1, skip_frames + 1)

    # Processing resolution (keep aspect ratio)
    proc_h = int(proc_width * src_h / src_w)

    # Output panel sizes
    panel_w = 640
    panel_h = int(panel_w * src_h / src_w)
    if render_3d:
        out_frame_w = panel_w * 3
    else:
        out_frame_w = panel_w * 2
    out_frame_h = panel_h + 50    # + HUD bar

    print("=" * 60)
    print("  DRONE SWARM VIDEO PIPELINE")
    print("=" * 60)
    print(f"  Input:       {video_path}")
    print(f"  Resolution:  {src_w}x{src_h} @ {src_fps:.1f} fps")
    print(f"  Frames:      {total_fr}")
    print(f"  Drones:      {n_drones}")
    print(f"  Proc size:   {proc_width}x{proc_h}")
    print(f"  Motion steps:{motion_steps}  Depth every:{depth_every}")
    print(f"  RL:          {'YES — ' + str(rl_checkpoint) if rl_checkpoint else 'No (physics)'}")
    print(f"  3D panel:    {'Yes' if render_3d else 'No'}")
    print(f"  Output:      {out_path}")
    print("=" * 60)

    # ── load RL if requested ─────────────────────────────────────────────────
    if rl_checkpoint:
        _load_rl(rl_checkpoint)

    # ── preload segmentation model so we don't load mid-loop ────────────────
    _load_segmentation()
    if depth_every > 0:
        _load_midas()

    # ── output video writer ──────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps,
                             (out_frame_w, out_frame_h))

    # ── CSV header ───────────────────────────────────────────────────────────
    csv_lines = ["frame,drone_id,x_norm,y_norm,z_depth\n"]

    # ── state ────────────────────────────────────────────────────────────────
    positions     = None    # (N, 2)  normalised
    depth_cache   = None    # last depth map at proc resolution
    prev_targets  = None    # for smoothing

    frame_times = []
    frame_idx   = 0
    written     = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or (max_frames > 0 and frame_idx >= max_frames):
            break

        # skip frames
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            continue

        t0 = time.perf_counter()

        # ── resize for processing ────────────────────────────────────────────
        proc_frame = cv2.resize(frame_bgr, (proc_width, proc_h))

        # ── segmentation ─────────────────────────────────────────────────────
        try:
            mask = segment_frame(proc_frame)
        except Exception as e:
            print(f"  [frame {frame_idx}] Segmentation error: {e} — skipping")
            frame_idx += 1
            continue

        has_person = mask.any()

        if has_person:
            try:
                targets_raw = sample_targets(mask, n_drones)   # (N, 2) normalised
            except Exception:
                has_person = False

        # ── depth every N frames ─────────────────────────────────────────────
        if has_person and depth_every > 0 and frame_idx % depth_every == 0:
            try:
                depth_cache = depth_frame(proc_frame)
            except Exception as e:
                print(f"  [frame {frame_idx}] Depth error: {e}")

        # ── initialise positions on first valid frame ────────────────────────
        if positions is None:
            if not has_person:
                frame_idx += 1
                continue
            positions = targets_raw.copy()
            prev_targets = targets_raw.copy()

        if has_person:
            # Smooth target positions with prev_targets (EMA damping)
            alpha_smooth = 0.65
            targets_smooth = alpha_smooth * targets_raw + (1 - alpha_smooth) * prev_targets
            prev_targets = targets_smooth.copy()

            # Hungarian temporal reassignment
            targets_assigned = hungarian_reassign(positions, targets_smooth)
        else:
            # No person detected — drones hold last targets
            targets_assigned = prev_targets if prev_targets is not None else positions

        # ── motion: physics or RL ────────────────────────────────────────────
        if rl_checkpoint and _rl_ctrl is not None:
            for _ in range(motion_steps):
                # Build 3D positions (z from depth cache or zeros)
                if depth_cache is not None:
                    px_tmp, py_tmp = _norm_to_pixel(positions, proc_h, proc_width)
                    z_vals = depth_cache[py_tmp, px_tmp].astype(np.float32) * 3.0
                else:
                    z_vals = np.zeros(n_drones, np.float32)
                pos3d     = np.column_stack([positions, z_vals])
                tgt3d     = np.column_stack([targets_assigned,
                                             np.zeros(n_drones, np.float32)])
                try:
                    new3d     = _rl_ctrl.step(pos3d, tgt3d)
                    positions = new3d[:, :2].astype(np.float32)
                except Exception:
                    positions = physics_step(positions, targets_assigned,
                                             step_size, rep_strength, rep_radius)
        else:
            for _ in range(motion_steps):
                positions = physics_step(positions, targets_assigned,
                                         step_size, rep_strength, rep_radius)

        # Clamp positions to reasonable bounds to prevent runaway drift
        positions = np.clip(positions, -8.0, 8.0)

        # ── metrics ──────────────────────────────────────────────────────────
        mean_err = float(np.linalg.norm(positions - targets_assigned, axis=1).mean())

        # ── extract depth values at drone positions ──────────────────────────
        px_vis, py_vis = _norm_to_pixel(positions, proc_h, proc_width)
        if depth_cache is not None:
            depth_vals = depth_cache[py_vis, px_vis].astype(np.float32)
        else:
            depth_vals = None

        # ── render panels ────────────────────────────────────────────────────
        disp_frame = cv2.resize(frame_bgr, (panel_w, panel_h))
        proc_disp  = cv2.resize(proc_frame, (panel_w, panel_h))
        scale_x = panel_w / proc_width
        scale_y = panel_h / proc_h

        # positions scaled for panel
        pos_for_render = positions.copy()   # normalised coords — render_overlay handles inverse

        overlay = render_overlay(proc_disp, pos_for_render, 
                                 cv2.resize(depth_cache, (panel_w, panel_h)) if depth_cache is not None else None,
                                 dot_radius=dot_radius, alpha=0.6)

        # HUD
        fps_est = 1.0 / max(1e-3, np.mean(frame_times[-10:])) if frame_times else 0.0
        hud = make_hud(frame_idx, n_drones, mean_err, fps_est,
                       rl_checkpoint is not None, 50, out_frame_w)

        if render_3d:
            panel3d = render_3d_axes(positions, depth_vals, (panel_h, panel_w))
            body = np.hstack([disp_frame, overlay, panel3d])
        else:
            body = np.hstack([disp_frame, overlay])

        out_frame = np.vstack([hud, body])
        if out_frame.shape[:2] != (out_frame_h, out_frame_w):
            out_frame = cv2.resize(out_frame, (out_frame_w, out_frame_h))

        writer.write(out_frame)
        written += 1

        # ── CSV ──────────────────────────────────────────────────────────────
        for i in range(n_drones):
            z = float(depth_vals[i]) if depth_vals is not None else 0.0
            csv_lines.append(f"{frame_idx},{i},{positions[i,0]:.4f},"
                             f"{positions[i,1]:.4f},{z:.4f}\n")

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        fps_now = 1.0 / (t1 - t0)
        pct = 100 * frame_idx / max(total_fr, 1)
        eta = (total_fr - frame_idx) * np.mean(frame_times[-20:])
        print(f"  [{frame_idx:04d}/{total_fr}] {pct:5.1f}%  "
              f"fps={fps_now:5.1f}  err={mean_err:.3f}  ETA={eta:.0f}s")

        frame_idx += 1

    cap.release()
    writer.release()

    # ── write CSV ─────────────────────────────────────────────────────────────
    with open(csv_path, "w") as f:
        f.writelines(csv_lines)

    print()
    print("=" * 60)
    print(f"  [DONE] {written} frames written")
    print(f"  Video:  {out_path}")
    print(f"  CSV:    {csv_path}  ({len(csv_lines)-1} rows)")
    print("=" * 60)
    return out_path, csv_path


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Drone Swarm Video Pipeline")
    p.add_argument("--video",      required=True,
                   help="Path to input video (MP4, AVI, …)")
    p.add_argument("--output",     default="output",
                   help="Output directory  [default: output/]")
    p.add_argument("--drones",     type=int,   default=300,
                   help="Number of drones  [default: 300]")
    p.add_argument("--proc-width", type=int,   default=640,
                   help="Processing width in pixels  [default: 640]")
    p.add_argument("--motion-steps", type=int, default=6,
                   help="Physics/RL steps per video frame  [default: 6]")
    p.add_argument("--depth-every",  type=int, default=5,
                   help="Run MiDaS every N frames (0=disable)  [default: 5]")
    p.add_argument("--rl-checkpoint", default=None,
                   help="Path to SB3 PPO checkpoint (no .zip)")
    p.add_argument("--render-3d",  action="store_true",
                   help="Add a 3D matplotlib panel (slower)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Process only first N frames (0=all)")
    p.add_argument("--skip-frames",type=int, default=0,
                   help="Skip every N frames for speed (0=none)")
    p.add_argument("--dot-radius", type=int, default=5,
                   help="Drone dot radius in pixels  [default: 5]")
    p.add_argument("--step-size",  type=float, default=0.15,
                   help="Physics attraction strength  [default: 0.15]")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        video_path    = args.video,
        output_dir    = args.output,
        n_drones      = args.drones,
        proc_width    = args.proc_width,
        motion_steps  = args.motion_steps,
        depth_every   = args.depth_every,
        rl_checkpoint = args.rl_checkpoint,
        render_3d     = args.render_3d,
        max_frames    = args.max_frames,
        skip_frames   = args.skip_frames,
        dot_radius    = args.dot_radius,
        step_size     = args.step_size,
    )
