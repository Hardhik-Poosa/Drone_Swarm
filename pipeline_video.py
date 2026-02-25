"""
pipeline_video.py  -  Drone Swarm Video Formation Generator  (v2)
=================================================================
Process a video file and produce an output video where a swarm of drones
traces the person / object silhouette in every frame, smoothly following motion.

Key design (v2)
---------------
  * PIXEL-SPACE coordinates throughout — no broken normalization
  * Lerp-based smooth tracking per frame — drones glide toward targets
  * NO physics repulsion — repulsion scatters dense formations
  * Hungarian assignment each frame for temporal consistency
  * MiDaS depth every N frames — colours the drone dots

Usage
-----
    python pipeline_video.py --video input_images/video.mp4
    python pipeline_video.py --video input_images/video.mp4 \
        --drones 300 --depth-every 5 --output output/
"""

import os, sys, argparse, time, math, warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

_seg_model  = None
_seg_tfm    = None
_midas_model = None
_midas_tfm  = None

PERSON_CLASS = 15


# ------------------------------------------------------------------
# Model loaders
# ------------------------------------------------------------------

def _load_seg():
    global _seg_model, _seg_tfm
    if _seg_model is not None:
        return
    print("[load] DeepLabV3-ResNet101 ...")
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
    w = DeepLabV3_ResNet101_Weights.DEFAULT
    _seg_model = tvm.segmentation.deeplabv3_resnet101(weights=w).eval()
    _seg_tfm = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print("[load] Segmentation ready.")


def _load_midas():
    global _midas_model, _midas_tfm
    if _midas_model is not None:
        return
    print("[load] MiDaS DPT-Large ...")
    import torch
    _midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True).eval()
    _midas_tfm   = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
    print("[load] Depth model ready.")


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------

def segment_frame(bgr: np.ndarray) -> np.ndarray:
    """Returns (H, W) bool mask at same size as bgr."""
    import torch
    from PIL import Image as PIL

    _load_seg()
    h, w = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil  = PIL.fromarray(rgb).resize((512, 512))
    inp  = _seg_tfm(np.array(pil)).unsqueeze(0)
    with torch.no_grad():
        pred = _seg_model(inp)["out"][0].argmax(0).numpy()
    mask512 = (pred == PERSON_CLASS).astype(np.uint8)
    mask    = cv2.resize(mask512, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m8   = mask.astype(np.uint8) * 255
    m8   = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k, iterations=3)
    m8   = cv2.morphologyEx(m8, cv2.MORPH_OPEN,  k, iterations=1)
    return m8.astype(bool)


# ------------------------------------------------------------------
# Depth
# ------------------------------------------------------------------

def get_depth(bgr: np.ndarray) -> np.ndarray:
    """Returns float32 [0..1] depth map at same size as bgr."""
    import torch

    _load_midas()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = _midas_tfm(rgb)
    if inp.dim() == 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        pred = _midas_model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
    d = pred.cpu().numpy().astype(np.float32)
    d -= d.min()
    if d.max() > 0:
        d /= d.max()
    return cv2.GaussianBlur(d, (11, 11), 0)


# ------------------------------------------------------------------
# Target sampling  (PIXEL space)
# ------------------------------------------------------------------

def _poisson_disk_px(mask: np.ndarray, n: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    area   = float(len(xs))
    cell   = max(1.0, math.sqrt(area / max(n, 1)) * 0.5)
    min_sq = cell ** 2
    rng    = np.random.default_rng(42)
    order  = rng.permutation(len(xs))
    xs_, ys_ = xs[order], ys[order]
    accepted, grid = [], {}

    def _key(x, y):
        return (int(x // cell), int(y // cell))

    for xi, yi in zip(xs_, ys_):
        cx, cy = _key(xi, yi)
        ok = True
        for dx in range(-2, 3):
            if not ok:
                break
            for dy in range(-2, 3):
                for nx_, ny_ in grid.get((cx+dx, cy+dy), []):
                    if (xi - nx_)**2 + (yi - ny_)**2 < min_sq:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            accepted.append((float(xi), float(yi)))
            grid.setdefault(_key(xi, yi), []).append((xi, yi))
            if len(accepted) >= n:
                break

    if len(accepted) < n:
        idx = rng.choice(len(xs), size=n - len(accepted), replace=True)
        for i in idx:
            accepted.append((float(xs[i]), float(ys[i])))

    return np.array(accepted, dtype=np.float32)  # (n, 2): col, row


def _outline_px(mask: np.ndarray, n: int) -> np.ndarray:
    cnts, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts or n == 0:
        return np.zeros((0, 2), np.float32)
    cnt    = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    diffs  = np.diff(cnt, axis=0)
    arclen = np.r_[0, np.cumsum(np.linalg.norm(diffs, axis=1))]
    total  = arclen[-1]
    if total < 1:
        return np.zeros((0, 2), np.float32)
    ts = np.linspace(0, total, n, endpoint=False)
    xi = np.interp(ts, arclen, cnt[:, 0])
    yi = np.interp(ts, arclen, cnt[:, 1])
    return np.column_stack([xi, yi]).astype(np.float32)


def sample_targets_px(mask: np.ndarray, n: int) -> np.ndarray:
    """Return (n, 2) target pixel coords: col (X) and row (Y)."""
    n_out = max(0, n // 5)
    n_in  = n - n_out
    pts_in  = _poisson_disk_px(mask, n_in)
    pts_out = _outline_px(mask, n_out)
    if len(pts_out):
        pts = np.vstack([pts_in, pts_out])
    else:
        pts = pts_in
    if len(pts) < n:
        pts = np.vstack([pts, pts[:(n - len(pts))]])
    return pts[:n].astype(np.float32)


# ------------------------------------------------------------------
# Hungarian assignment (pixel space)
# ------------------------------------------------------------------

def hungarian_assign(positions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    cost    = np.linalg.norm(positions[:, None, :] - targets[None, :, :], axis=2)
    _, col  = linear_sum_assignment(cost)
    return targets[col]


# ------------------------------------------------------------------
# Lerp step
# ------------------------------------------------------------------

def lerp_step(positions: np.ndarray, targets: np.ndarray,
              alpha: float = 0.30) -> np.ndarray:
    return (positions + alpha * (targets - positions)).astype(np.float32)


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

def _depth_colour(d: float) -> tuple:
    d  = float(np.clip(d, 0.0, 1.0))
    r  = int(np.clip(d * 2 * 255, 0, 255))
    g  = int(np.clip((1.0 - abs(d - 0.5) * 2) * 255, 0, 255))
    b  = int(np.clip((1.0 - d) * 2 * 255, 0, 255))
    return (b, g, r)


def render_frame(bgr: np.ndarray,
                 pos_px: np.ndarray,
                 tgt_px=None,
                 depth=None,
                 dot_r: int = 5,
                 show_targets: bool = False) -> np.ndarray:
    h, w    = bgr.shape[:2]
    overlay = bgr.copy()
    if show_targets and tgt_px is not None:
        for i in range(len(tgt_px)):
            tx = int(np.clip(tgt_px[i, 0], 0, w - 1))
            ty = int(np.clip(tgt_px[i, 1], 0, h - 1))
            cv2.circle(overlay, (tx, ty), max(1, dot_r - 2), (60, 60, 60), -1, cv2.LINE_AA)
    for i in range(len(pos_px)):
        cx = int(np.clip(pos_px[i, 0], 0, w - 1))
        cy = int(np.clip(pos_px[i, 1], 0, h - 1))
        if depth is not None:
            col = _depth_colour(float(depth[cy, cx]))
        else:
            # cyan to orange gradient top-to-bottom
            t   = cy / max(h - 1, 1)
            col = (int(255 * (1 - t)), int(180 + 70 * (1 - t)), int(255 * t))
        cv2.circle(overlay, (cx, cy), dot_r,     col,          -1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), dot_r + 1, (220, 220, 220), 1, cv2.LINE_AA)
    return cv2.addWeighted(bgr, 0.35, overlay, 0.65, 0)


def make_hud(w: int, fi: int, tot: int, n: int,
             err: float, fps: float, has_d: bool) -> np.ndarray:
    bar  = np.zeros((44, w, 3), np.uint8)
    pb_w = int(w * fi / max(tot, 1))
    cv2.rectangle(bar, (0, 40), (pb_w, 43), (0, 180, 80), -1)
    mode = "Seg+Depth" if has_d else "Seg-only"
    txt  = (f"  Frame {fi:04d}/{tot}  Drones:{n}  "
            f"Err:{err:.0f}px  {mode}  {fps:.1f}fps")
    cv2.putText(bar, txt, (4, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 230, 160), 1, cv2.LINE_AA)
    return bar


def render_3d_panel(pos_px, depth, proc_h, proc_w, pw, ph):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cx = pos_px[:, 0] / max(proc_w - 1, 1)
    cy = 1.0 - pos_px[:, 1] / max(proc_h - 1, 1)
    if depth is not None:
        iy = pos_px[:, 1].clip(0, proc_h - 1).astype(int)
        ix = pos_px[:, 0].clip(0, proc_w - 1).astype(int)
        cz = depth[iy, ix]
    else:
        cz = np.zeros(len(cx))
    fig = plt.figure(figsize=(pw / 100, ph / 100), dpi=100)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#080808")
    fig.patch.set_facecolor("#080808")
    ax.scatter(cx, cy, cz * 3, c=cz, cmap="plasma", s=5, alpha=0.85, edgecolors="none")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#444")
    ax.tick_params(colors="white", labelsize=4)
    ax.set_title("3D view", color="white", fontsize=7, pad=2)
    ax.view_init(elev=25, azim=-55)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(ph, pw, 4)
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def process_video(
    video_path: str,
    output_dir: str    = "output",
    n_drones: int      = 300,
    proc_width: int    = 640,
    lerp_alpha: float  = 0.30,
    depth_every: int   = 6,
    render_3d: bool    = False,
    show_targets: bool = False,
    max_frames: int    = 0,
    dot_radius: int    = 5,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem     = Path(video_path).stem
    out_path = str(Path(output_dir) / f"{stem}_swarm.mp4")
    csv_path = str(Path(output_dir) / f"{stem}_positions.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    proc_h = int(round(proc_width * src_h / src_w))
    proc_h = proc_h + (proc_h % 2)   # make even

    panel_w, panel_h = proc_width, proc_h
    hud_h    = 44
    n_panels = 3 if render_3d else 2
    out_w    = panel_w * n_panels
    out_h    = panel_h + hud_h

    print("=" * 64)
    print("  DRONE SWARM VIDEO PIPELINE  v2  (pixel-space tracking)")
    print("=" * 64)
    print(f"  Input:       {video_path}")
    print(f"  Source:      {src_w}x{src_h} @ {src_fps:.1f} fps  ({total} frames)")
    print(f"  Proc size:   {proc_width}x{proc_h}")
    print(f"  Drones:      {n_drones}")
    print(f"  Lerp alpha:  {lerp_alpha}")
    if depth_every > 0:
        print(f"  Depth:       every {depth_every} frames")
    else:
        print(f"  Depth:       disabled")
    print(f"  Output:      {out_path}")
    print("=" * 64)

    _load_seg()
    if depth_every > 0:
        _load_midas()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, src_fps, (out_w, out_h))

    csv_lines = ["frame,drone_id,px_col,px_row,depth\n"]

    positions    = None
    prev_targets = None
    depth_cache  = None
    no_pers      = 0
    frame_times  = []

    for frame_idx in range(total):
        ret, raw = cap.read()
        if not ret:
            break

        t0   = time.perf_counter()
        proc = cv2.resize(raw, (proc_width, proc_h))

        # Segmentation
        try:
            mask = segment_frame(proc)
        except Exception as e:
            print(f"  [{frame_idx:04d}] seg err: {e}")
            mask = np.zeros((proc_h, proc_width), bool)

        has_person = bool(mask.sum() > 200)

        if has_person:
            try:
                tgts_raw = sample_targets_px(mask, n_drones)
                no_pers  = 0
            except Exception as e:
                print(f"  [{frame_idx:04d}] sample err: {e}")
                has_person = False

        if not has_person:
            no_pers += 1
            if prev_targets is not None:
                cx0, cy0 = proc_width / 2, proc_h / 2
                drift    = 0.01 * min(no_pers, 50) / 50
                tgts_raw = (prev_targets * (1 - drift)
                            + np.array([[cx0, cy0]], dtype=np.float32) * drift)
            else:
                tgts_raw = np.tile([[proc_width / 2.0, proc_h / 2.0]],
                                   (n_drones, 1)).astype(np.float32)

        # EMA smooth targets
        if prev_targets is None:
            tgt_smooth    = tgts_raw.copy()
        else:
            tgt_smooth    = 0.7 * tgts_raw + 0.3 * prev_targets
        prev_targets = tgt_smooth.copy()

        # Initialise positions at first frame's targets
        if positions is None:
            positions = tgt_smooth.copy()

        # Hungarian then lerp
        assigned  = hungarian_assign(positions, tgt_smooth)
        positions = lerp_step(positions, assigned, lerp_alpha)
        positions[:, 0] = np.clip(positions[:, 0], 0, proc_width - 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, proc_h - 1)

        # Depth
        if depth_every > 0 and frame_idx % depth_every == 0:
            try:
                depth_cache = get_depth(proc)
            except Exception as e:
                print(f"  [{frame_idx:04d}] depth err: {e}")

        # Metrics
        err_px = float(np.linalg.norm(positions - assigned, axis=1).mean())

        # Render
        overlay = render_frame(proc, positions, assigned,
                               depth_cache, dot_radius, show_targets)
        t1 = time.perf_counter()
        fps_now = 1.0 / max(t1 - t0, 1e-4)
        hud = make_hud(out_w, frame_idx, total, n_drones,
                       err_px, fps_now, depth_cache is not None)

        panels = [proc, overlay]
        if render_3d:
            panels.append(render_3d_panel(positions, depth_cache,
                                          proc_h, proc_width, panel_w, panel_h))
        body = np.hstack(panels)
        if body.shape[1] != out_w or body.shape[0] != panel_h:
            body = cv2.resize(body, (out_w, panel_h))
        out_frame = np.vstack([hud, body])
        writer.write(out_frame)

        # CSV
        for i in range(n_drones):
            cx_ = int(np.clip(positions[i, 0], 0, proc_width - 1))
            cy_ = int(np.clip(positions[i, 1], 0, proc_h - 1))
            dv  = float(depth_cache[cy_, cx_]) if depth_cache is not None else 0.0
            csv_lines.append(f"{frame_idx},{i},{cx_},{cy_},{dv:.4f}\n")

        frame_times.append(t1 - t0)
        avg_fps = 1.0 / (np.mean(frame_times[-15:]) + 1e-9)
        eta     = (total - frame_idx - 1) * np.mean(frame_times[-15:])
        pflag   = "person" if has_person else "no-person"
        print(f"  [{frame_idx:04d}/{total}] {100*(frame_idx+1)/total:5.1f}%  "
              f"fps={avg_fps:5.1f}  err={err_px:5.0f}px  {pflag}  ETA={eta:.0f}s")

    cap.release()
    writer.release()
    with open(csv_path, "w") as f:
        f.writelines(csv_lines)

    written = frame_idx + 1
    print()
    print("=" * 64)
    print(f"  [DONE]  {written} frames")
    print(f"  Video:  {out_path}")
    print(f"  CSV:    {csv_path}")
    print("=" * 64)
    return out_path, csv_path


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _args():
    p = argparse.ArgumentParser(description="Drone Swarm Video v2")
    p.add_argument("--video",        required=True)
    p.add_argument("--output",       default="output")
    p.add_argument("--drones",       type=int,   default=300)
    p.add_argument("--proc-width",   type=int,   default=640)
    p.add_argument("--lerp-alpha",   type=float, default=0.30)
    p.add_argument("--depth-every",  type=int,   default=6)
    p.add_argument("--render-3d",    action="store_true")
    p.add_argument("--show-targets", action="store_true")
    p.add_argument("--max-frames",   type=int,   default=0)
    p.add_argument("--dot-radius",   type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    process_video(
        video_path   = a.video,
        output_dir   = a.output,
        n_drones     = a.drones,
        proc_width   = a.proc_width,
        lerp_alpha   = a.lerp_alpha,
        depth_every  = a.depth_every,
        render_3d    = a.render_3d,
        show_targets = a.show_targets,
        max_frames   = a.max_frames,
        dot_radius   = a.dot_radius,
    )
