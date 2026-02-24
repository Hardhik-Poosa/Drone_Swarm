import cv2
import numpy as np

def image_to_outline(image_path, n_drones=50):

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize for consistency
    img = cv2.resize(img, (400, 400))

    # Edge detection
    edges = cv2.Canny(img, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError("No contours found in image.")

    # Use largest contour
    contour = max(contours, key=lambda x: len(x))

    contour = contour.squeeze()

    # Sample N evenly spaced points
    indices = np.linspace(0, len(contour)-1, n_drones).astype(int)
    sampled = contour[indices]

    # Normalize to centered coordinates
    sampled = sampled.astype(np.float32)
    sampled -= np.mean(sampled, axis=0)
    sampled /= np.max(np.abs(sampled))

    return sampled