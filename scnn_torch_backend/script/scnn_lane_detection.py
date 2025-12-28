from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# Import from scnn_torch
SCNN_ROOT = Path(__file__).resolve().parent.parent.parent
# For interactive testing only
# SCNN_ROOT = Path('/home/yi-chen/python_ws')
sys.path.insert(0, str(SCNN_ROOT))

from scnn_torch.model import SCNN
from scnn_torch.utils import resize_seg_pred, visualize_lanes


# === Step 1: Configuration ===
img_path = './test/image_000.png'  # Provide the path to your image
checkpoint_path = SCNN_ROOT / 'scnn_torch' / 'checkpoints' / 'best.pth'
target_height = 288  # Must be divisible by 8

# ImageNet normalization constants
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# === Step 2: Load and preprocess image ===
img_pil = Image.open(img_path).convert('RGB')
original_size = (img_pil.height, img_pil.width)  # (H, W)

# Calculate target width preserving aspect ratio, divisible by 8
target_width = round(original_size[1] * target_height / original_size[0] / 8) * 8

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((target_height, target_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Apply transformations
input_tensor = transform(img_pil).unsqueeze(0)  # Shape: [1, 3, H, W]


# === Step 3: Load pretrained SCNN model ===
model = SCNN(ms_ks=9, pretrained=False)

print(f'Loading checkpoint: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['net'])
print(f"Loaded from iteration {checkpoint.get('iteration', 'unknown')}")

model.eval()


# === Step 4: Run inference ===
with torch.no_grad():
    seg_pred, exist_pred = model(input_tensor)

    # Convert logits to probabilities
    seg_pred = F.softmax(seg_pred, dim=1)
    exist_pred = torch.sigmoid(exist_pred)

# Convert to numpy
seg_pred = seg_pred.squeeze(0).cpu().numpy()  # Shape: [5, H, W]
exist_pred = exist_pred.squeeze(0).cpu().numpy()  # Shape: [4]


# === Step 5: Post-process ===
# Resize seg_pred back to original image size
seg_pred_resized = resize_seg_pred(seg_pred, original_size)

# Print lane existence probabilities
print(f"Lane existence probabilities: {[f'{p:.2f}' for p in exist_pred]}")


# === Step 6: Visualize with matplotlib ===
# Convert PIL image to numpy array for visualization
img_np = np.array(img_pil)

# Get overlay and lane mask using visualize_lanes
img_overlay, lane_img = visualize_lanes(img_np, seg_pred_resized, exist_pred, threshold=0.75)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(3, 1, 1)
plt.imshow(img_np)
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(img_overlay)
plt.title('Overlay (Original + Lanes)')
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(lane_img)
plt.title('Lane Mask')
plt.axis('off')

plt.tight_layout()
plt.show()
