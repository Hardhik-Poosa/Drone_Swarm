import torch
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------
# Load DeepLabV3 Properly (Modern API)
# ---------------------------------------
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = models.segmentation.deeplabv3_resnet101(weights=weights)
model.eval()

# COCO class index for 'person'
PERSON_CLASS = 15

# ---------------------------------------
# Image Preprocessing
# ---------------------------------------
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def image_to_semantic_outline(image_path, n_drones=300, scale_factor=6):

    # ---------------------------------------
    # Load Image
    # ---------------------------------------
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # ---------------------------------------
    # Run Segmentation
    # ---------------------------------------
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    segmentation = output.argmax(0).byte().cpu().numpy()

    # ---------------------------------------
    # Extract Person Mask
    # ---------------------------------------
    mask = (segmentation == PERSON_CLASS).astype(np.uint8) * 255

    # Increase resolution for smoother contour
    mask = cv2.resize(mask, (2000, 2000))

    # ---------------------------------------
    # Extract Contours
    # ---------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        raise ValueError("No person detected in image.")

    # Use largest contour
    contour = max(contours, key=lambda x: len(x))
    contour = contour.squeeze().astype(np.float32)

    if len(contour.shape) != 2:
        raise ValueError("Contour extraction failed.")

    # ---------------------------------------
    # Proper Arc-Length Based Sampling
    # ---------------------------------------

    # Close contour loop
    contour_closed = np.vstack([contour, contour[0]])

    # Compute segment distances
    segment_lengths = np.sqrt(np.sum(np.diff(contour_closed, axis=0)**2, axis=1))

    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)

    total_length = cumulative_lengths[-1]

    # Evenly spaced distances along contour
    even_distances = np.linspace(0, total_length, n_drones)

    sampled = []

    for d in even_distances:
        idx = np.searchsorted(cumulative_lengths, d)
        idx = min(idx, len(contour_closed)-1)
        sampled.append(contour_closed[idx])

    sampled = np.array(sampled, dtype=np.float32)

    # ---------------------------------------
    # Center + Flip Y-axis (fix orientation)
    # ---------------------------------------
    sampled -= np.mean(sampled, axis=0)

    sampled[:, 1] = -sampled[:, 1]

    # ---------------------------------------
    # Normalize
    # ---------------------------------------
    max_val = np.max(np.abs(sampled))
    if max_val != 0:
        sampled /= max_val

    # ---------------------------------------
    # Scale for Visibility
    # ---------------------------------------
    sampled *= scale_factor

    return sampled