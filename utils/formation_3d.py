import numpy as np
from utils.depth_to_3d import get_depth_map
from scipy.spatial import cKDTree


def lift_to_true_3d(points_2d, image_path, height_scale=8, layers=4):

    depth_map = get_depth_map(image_path)

    h, w = depth_map.shape

    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()

    base_points = []

    for x, y in points_2d:

        px = int((x - x_min) / (x_max - x_min) * (w - 1))
        py = int((y - y_min) / (y_max - y_min) * (h - 1))

        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)

        z = depth_map[py, px] * height_scale

        base_points.append([x, y, z])

    base_points = np.array(base_points)

    # ---------- Uniform 3D Thickness ----------
    volumetric_points = []

    for x, y, z in base_points:
        for i in range(layers):
            volumetric_points.append([
                x,
                y,
                z - 0.6 + i*(1.2/layers)
            ])

    volumetric_points = np.array(volumetric_points)

    # ---------- Remove Outliers (Statistical Cleaning) ----------
    tree = cKDTree(volumetric_points)
    distances, _ = tree.query(volumetric_points, k=6)

    mean_dist = distances[:, 1:].mean(axis=1)
    threshold = np.mean(mean_dist) + 2*np.std(mean_dist)

    clean_points = volumetric_points[mean_dist < threshold]

    return clean_points