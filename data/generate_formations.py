import numpy as np

# -----------------------
# CONFIG
# -----------------------
N_DRONES = 10
NUM_SAMPLES = 10000
FORMATION_TYPES = ["line", "circle", "v_shape", "grid"]

# -----------------------
# FORMATION GENERATORS
# -----------------------

def line_formation(n, spacing=1.0):
    x = np.linspace(-spacing * (n // 2), spacing * (n // 2), n)
    y = np.zeros(n)
    return np.column_stack((x, y)).flatten()

def circle_formation(n, radius=5.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y)).flatten()

def v_shape_formation(n, spacing=1.0):
    half = n // 2
    left_x = np.linspace(-half, -1, half)
    left_y = np.linspace(0, half, half)

    right_x = np.linspace(1, half, half)
    right_y = np.linspace(half, 0, half)

    x = np.concatenate((left_x, right_x))
    y = np.concatenate((left_y, right_y))

    return np.column_stack((x * spacing, y * spacing)).flatten()

def grid_formation(n, spacing=1.5):
    grid_size = int(np.ceil(np.sqrt(n)))
    points = []

    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n:
                points.append([i * spacing, j * spacing])

    return np.array(points).flatten()

# -----------------------
# DATASET GENERATION
# -----------------------

def generate_dataset():
    data = []

    for _ in range(NUM_SAMPLES):
        formation_type = np.random.choice(FORMATION_TYPES)

        if formation_type == "line":
            formation = line_formation(N_DRONES)

        elif formation_type == "circle":
            formation = circle_formation(N_DRONES)

        elif formation_type == "v_shape":
            formation = v_shape_formation(N_DRONES)

        elif formation_type == "grid":
            formation = grid_formation(N_DRONES)

        # Add small noise to increase diversity
        noise = np.random.normal(0, 0.05, size=formation.shape)
        formation += noise

        data.append(formation)

    data = np.array(data)
    np.save("formations.npy", data)

    print(f"Saved dataset with shape: {data.shape}")
    print("Each formation vector size:", data.shape[1])

if __name__ == "__main__":
    generate_dataset()
