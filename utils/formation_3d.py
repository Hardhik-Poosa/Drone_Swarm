"""
formation_3d.py – Lift 2D drone waypoints to true 3D using MiDaS depth.

Design principle
----------------
Each 2D silhouette point maps to exactly ONE 3D point.
  X, Y  ->  preserved from silhouette  (front view = perfect person shape)
  Z     ->  MiDaS depth value          (natural depth, closer parts pop forward)

No layer-stacking: stacking creates thick rectangular blobs that look
nothing like the person when viewed at any angle.
"""

import numpy as np
from utils.depth_to_3d import get_depth_map


def lift_to_true_3d(points_2d: np.ndarray,
                    image_path: str,
                    height_scale: float = 3.0,
                    layers: int = 1) -> np.ndarray:
    """
    Convert (N, 2) silhouette points to (N, 3) 3D drone positions.

    X, Y are preserved exactly – the front-view projection looks
    identical to the 2D silhouette.  Z comes from MiDaS depth so
    drones closest to the camera sit at higher Z values.

    Parameters
    ----------
    points_2d    : (N, 2) float32  world-space XY from image_to_semantic_outline
    image_path   : str             path to input image (for MiDaS depth)
    height_scale : float           multiplier on normalised depth  (default 3.0)
    layers       : int             kept for API compatibility, always 1 now
    """
    depth_map = get_depth_map(image_path)   # H x W, values in ~[0, 1]
    h, w = depth_map.shape

    pts = points_2d.astype(np.float64)
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    result = np.empty((len(pts), 3), dtype=np.float32)

    for i, (x, y) in enumerate(pts):
        # Map world XY -> pixel.  Y is flipped (world Y-up, image Y-down).
        px = int(np.clip((x - x_min) / x_range * (w - 1), 0, w - 1))
        py = int(np.clip((1.0 - (y - y_min) / y_range) * (h - 1), 0, h - 1))

        z = float(depth_map[py, px]) * height_scale
        result[i] = [x, y, z]

    return result