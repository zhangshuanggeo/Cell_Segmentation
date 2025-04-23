from cellpose import models
import cv2
import numpy as np

# Load grayscale image
img = cv2.imread("sample_data/Image 01_denoise.tiff", cv2.IMREAD_GRAYSCALE).astype(np.float32)
img0 = cv2.imread("sample_data/Image 01_original.tiff", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Run Cellpose
model = models.Cellpose(model_type='cyto')
masks_list, flows, styles, diams = model.eval([img], diameter=None, channels=[0, 0])

# Unpack the result for this single image
mask = masks_list[0]
from cellpose.utils import masks_to_outlines
from matplotlib import pyplot as plt

# Overlay mask outlines on original image
outlines = masks_to_outlines(mask)

# Make an overlay image
overlay = img.copy()
overlay[outlines > 0] = 255  # white outlines

overlay0 = img0.copy()
overlay0[outlines > 0] = 255  # white outlines

# Plot manually
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img0, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Original Segmentation")
plt.imshow(overlay0, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Denoised")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Denoised Segmentation")
plt.imshow(overlay, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
import cv2
from cellpose.utils import masks_to_outlines

outlines = masks_to_outlines(mask)

overlay_original = img.copy()
overlay_original[outlines > 0] = 255  # white outlines

overlay_denoised = img0.copy()
overlay_denoised[outlines > 0] = 255

cv2.imwrite("sample_data/segmentation_overlay_original.png", overlay_original.astype(np.uint8))
cv2.imwrite("sample_data/segmentation_overlay_denoised.png", overlay_denoised.astype(np.uint8))

cv2.imwrite("sample_data/segmentation_mask.tiff", mask.astype(np.uint16))  # 16-bit TIFF with cell labels
