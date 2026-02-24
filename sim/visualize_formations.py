import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("../data/formations.npy")

def visualize_formation(formation, title="Drone Formation"):
    coords = formation.reshape(-1, 2)
    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c="blue", s=60)
    for i in range(len(x)):
        plt.text(x[i] + 0.05, y[i] + 0.05, str(i), fontsize=9)

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# -----------------------
# SHOW EXAMPLES
# -----------------------

# Show 4 random formations
for i in range(4):
    idx = np.random.randint(0, len(data))
    visualize_formation(data[idx], title=f"Sample Formation {i+1}")
