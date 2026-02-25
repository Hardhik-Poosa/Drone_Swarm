import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Load DeepLabV3 once at import time
# ─────────────────────────────────────────────────────────────────────────────
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = models.segmentation.deeplabv3_resnet101(weights=weights)
model.eval()

PERSON_CLASS = 15   # COCO class index for 'person'

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────────────────────
# Poisson-disk sampler (fast dart-throwing inside a binary mask)
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_disk_sample_mask(mask_bool: np.ndarray,
                               n_points: int,
                               min_dist_ratio: float = 0.4) -> np.ndarray:
    """
    Sample *n_points* positions that are:
      - inside *mask_bool*  (H×W boolean array)
      - reasonably spread out (Poisson-disk style)

    Returns (n_points, 2) float32 in pixel coordinates (col, row).
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        raise ValueError("Mask is empty – no person detected.")

    # Estimate min pixel distance from desired drone count & mask area
    area        = len(xs)
    cell        = np.sqrt(area / n_points) * min_dist_ratio
    min_dist_sq = cell ** 2

    rng         = np.random.default_rng(0)
    accepted    = []
    grid        = {}                              # sparse grid for speed

    # Shuffle candidate pixels so we don't always pick top-left first
    order = rng.permutation(len(xs))
    xs_s, ys_s = xs[order], ys[order]

    def _cell_key(x, y):
        return (int(x // cell), int(y // cell))

    def _too_close(x, y):
        cx, cy = _cell_key(x, y)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nb = grid.get((cx + dx, cy + dy))
                if nb is not None:
                    if (nb[0] - x) ** 2 + (nb[1] - y) ** 2 < min_dist_sq:
                        return True
        return False

    for x, y in zip(xs_s, ys_s):
        if len(accepted) >= n_points:
            break
        if not _too_close(x, y):
            accepted.append([x, y])
            grid[_cell_key(x, y)] = (x, y)

    # If Poisson-disk gave fewer than needed, top-up with random mask points
    if len(accepted) < n_points:
        remaining = n_points - len(accepted)
        extra_idx = rng.choice(len(xs), size=remaining, replace=False)
        for i in extra_idx:
            accepted.append([xs[i], ys[i]])

    pts = np.array(accepted[:n_points], dtype=np.float32)
    return pts[:, [0, 1]]   # (col, row) = (x, y) in pixel space


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def image_to_semantic_outline(image_path: str,
                               n_drones: int = 300,
                               scale_factor: float = 6,
                               outline_weight: float = 0.25) -> np.ndarray:
    """
    Extract a precise drone formation from a person image.

    Strategy
    --------
    *  The body silhouette is FILLED uniformly with drones (Poisson-disk
       sampling) so every part of the person is represented.
    *  A fraction (`outline_weight`) of drones are placed on the contour
       edge so the silhouette boundary stays sharp and recognisable.
    *  Aspect ratio of the original image is preserved exactly.
    *  Y-axis is flipped to match standard 2-D physics coords (Y up).

    Parameters
    ----------
    image_path     : str    path to the image
    n_drones       : int    total number of drones
    scale_factor   : float  world-space half-extent of the formation
    outline_weight : float  fraction of drones on edge (0 = pure fill,
                            1 = pure outline). Default 0.25.

    Returns
    -------
    (n_drones, 2) float32 array of (x, y) world-space coordinates
    """

    # ── 1. Load + segment ────────────────────────────────────────────────────
    pil_img       = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    input_tensor  = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    seg = output.argmax(0).byte().cpu().numpy()     # 512×512 label map

    # ── 2. Build mask at high resolution (preserving aspect ratio) ──────────
    MASK_LONG = 1024        # long-edge resolution for precision
    if orig_w >= orig_h:
        mask_w = MASK_LONG
        mask_h = max(1, int(MASK_LONG * orig_h / orig_w))
    else:
        mask_h = MASK_LONG
        mask_w = max(1, int(MASK_LONG * orig_w / orig_h))

    # Resize 512×512 seg map → aspect-correct mask
    seg_resized = cv2.resize(
        (seg == PERSON_CLASS).astype(np.uint8) * 255,
        (mask_w, mask_h),
        interpolation=cv2.INTER_NEAREST
    )

    # Morphological cleanup: fill small holes, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    seg_resized = cv2.morphologyEx(seg_resized, cv2.MORPH_CLOSE, kernel)
    seg_resized = cv2.morphologyEx(seg_resized, cv2.MORPH_OPEN,  kernel)

    mask_bool = seg_resized > 127

    if not mask_bool.any():
        raise ValueError("No person detected in image.")

    # ── 3. Sample fill points (interior) ────────────────────────────────────
    n_fill    = max(1, n_drones - int(n_drones * outline_weight))
    fill_pts  = _poisson_disk_sample_mask(mask_bool, n_fill)   # (x, y) pixels

    # ── 4. Sample outline points (contour) ───────────────────────────────────
    n_outline = n_drones - len(fill_pts)
    outline_pts = np.empty((0, 2), dtype=np.float32)

    if n_outline > 0:
        contours, _ = cv2.findContours(
            seg_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            # Arc-length interpolated sampling on the largest contour
            contour = max(contours, key=cv2.contourArea).squeeze().astype(np.float32)
            if contour.ndim == 2 and len(contour) >= 2:
                closed   = np.vstack([contour, contour[0]])
                seg_lens = np.linalg.norm(np.diff(closed, axis=0), axis=1)
                cum_lens = np.concatenate([[0], np.cumsum(seg_lens)])
                total    = cum_lens[-1]
                targets  = np.linspace(0, total, n_outline, endpoint=False)

                pts_out  = []
                for t in targets:
                    idx = np.searchsorted(cum_lens, t, side='right') - 1
                    idx = np.clip(idx, 0, len(contour) - 1)
                    # Linear interpolation between vertices
                    seg_start = closed[idx]
                    seg_end   = closed[idx + 1]
                    seg_len   = seg_lens[idx]
                    if seg_len > 0:
                        alpha = (t - cum_lens[idx]) / seg_len
                    else:
                        alpha = 0.0
                    pts_out.append(seg_start + alpha * (seg_end - seg_start))

                outline_pts = np.array(pts_out, dtype=np.float32)  # (x, y) pixels

    # ── 5. Combine fill + outline ─────────────────────────────────────────────
    all_pts = np.vstack([fill_pts, outline_pts]) if len(outline_pts) else fill_pts

    # ── 6. Convert pixel → world coords, preserve aspect ratio ───────────────
    #   pixel x → world x,   pixel y → world y  (y will be flipped below)
    pts_x = all_pts[:, 0].astype(np.float64)
    pts_y = all_pts[:, 1].astype(np.float64)

    # Normalise so that the *longest* axis spans [-1, 1]
    cx, cy   = pts_x.mean(), pts_y.mean()
    pts_x   -= cx
    pts_y   -= cy

    max_range = max(np.ptp(pts_x), np.ptp(pts_y)) / 2.0 + 1e-9
    pts_x    /= max_range       # now in [-1, 1] (approx), aspect ratio kept
    pts_y    /= max_range

    # Flip Y so that "up in image" = "up in world"
    pts_y    = -pts_y

    # Scale to world units
    pts_x    *= scale_factor
    pts_y    *= scale_factor

    result = np.stack([pts_x, pts_y], axis=1).astype(np.float32)
    return result
