import torch

def collision_loss(coords, min_dist=0.5):
    """
    Penalize drones that are too close
    coords: (batch, N, 2)
    """
    loss = 0.0
    batch_size, N, _ = coords.shape

    for b in range(batch_size):
        for i in range(N):
            for j in range(i + 1, N):
                dist = torch.norm(coords[b, i] - coords[b, j])
                if dist < min_dist:
                    loss += (min_dist - dist) ** 2

    return loss / batch_size


def connectivity_loss(coords, max_dist=5.0):
    """
    Penalize drones that are too far apart
    """
    loss = 0.0
    batch_size, N, _ = coords.shape

    for b in range(batch_size):
        for i in range(N):
            for j in range(i + 1, N):
                dist = torch.norm(coords[b, i] - coords[b, j])
                if dist > max_dist:
                    loss += (dist - max_dist) ** 2

    return loss / batch_size
