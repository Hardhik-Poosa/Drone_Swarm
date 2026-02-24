import torch
import cv2
import numpy as np

# -------------------------------
# Load MiDaS model ONCE globally
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "DPT_Large"

midas = torch.hub.load(
    "intel-isl/MiDaS",
    model_type,
    trust_repo=True
)

midas.to(device)
midas.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms",
    trust_repo=True
)

transform = midas_transforms.dpt_transform


def get_depth_map(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_tensor = transform(img).to(device)

    # IMPORTANT FIX → DO NOT unsqueeze again
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize depth
    depth -= depth.min()
    if depth.max() > 0:
        depth /= depth.max()

    # Smooth depth to remove noise
    depth = cv2.GaussianBlur(depth, (9, 9), 0)

    return depth