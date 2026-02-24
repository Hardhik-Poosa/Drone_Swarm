import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Fix project root path
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.vae import VAE

# -------------------------------------------------
# Load trained model (FROM PROJECT ROOT)
# -------------------------------------------------
model_path = os.path.join(PROJECT_ROOT, "vae_model.pth")

model = VAE(input_dim=20, latent_dim=16)
model.load_state_dict(torch.load(model_path))
model.eval()

# -------------------------------------------------
# Generate and plot new formation
# -------------------------------------------------
def generate_and_plot():
    z = torch.randn(1, 16)

    with torch.no_grad():
        formation = model.decode(z).numpy().reshape(-1, 2)

    plt.figure(figsize=(5, 5))
    plt.scatter(formation[:, 0], formation[:, 1], s=60, c="blue")

    for i in range(len(formation)):
        plt.text(formation[i, 0] + 0.05, formation[i, 1] + 0.05, str(i))

    plt.title("Generated Drone Formation (VAE)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    generate_and_plot()
