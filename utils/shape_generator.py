import numpy as np

def generate_shape(shape, n_drones=10, distance=1.5):
    if shape == "line":
        x = np.linspace(
            -distance * (n_drones // 2),
            distance * (n_drones // 2),
            n_drones
        )
        y = np.zeros(n_drones)
        return np.column_stack((x, y))

    elif shape == "circle":
        radius = distance * n_drones / (2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.column_stack((x, y))

    elif shape == "v":
        half = n_drones // 2
        left = np.array([
            [-i * distance, i * distance] for i in range(half)
        ])
        right = np.array([
            [i * distance, i * distance] for i in range(half)
        ])
        return np.vstack((left, right))[:n_drones]

    elif shape == "grid":
        grid_size = int(np.ceil(np.sqrt(n_drones)))
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) < n_drones:
                    points.append([i * distance, j * distance])
        return np.array(points)

    else:
        raise ValueError("Unsupported shape")
