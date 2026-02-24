import os
import sys
import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.vae import VAE
from utils.metrics import collision_loss, connectivity_loss

# Load dataset
data_path = os.path.join(PROJECT_ROOT, "data", "formations.npy")
data = torch.tensor(np.load(data_path), dtype=torch.float32)

# Model
model = VAE(input_dim=20, latent_dim=16)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loss function
def loss_function(recon_x, x, mu, logvar):
    recon = torch.mean((recon_x - x) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    coords = recon_x.view(-1, 10, 2)

    coll = collision_loss(coords)
    conn = connectivity_loss(coords)

    return recon + kl + 0.5 * coll + 0.2 * conn

# Training
epochs = 150
for epoch in range(epochs):
    optimizer.zero_grad()

    recon_x, mu, logvar = model(data)
    loss = loss_function(recon_x, data, mu, logvar)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "vae_model.pth"))
print("Constraint-aware model saved.")
