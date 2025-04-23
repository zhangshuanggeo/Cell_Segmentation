import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from stardist.models import StarDist2D
from stardist import random_label_cmap
from csbdeep.utils import normalize
from skimage.color import rgb2gray # Import rgb2gray for channel conversion


# Load image
img = imread("sample_data/Image 01_original.tiff")

# Convert image to grayscale if it has more than 3 channels or is RGBA
if img.ndim == 3 and (img.shape[2] > 3 or img.shape[2] == 4): #check if image has more than 3 channels or is RGBA
    img = rgb2gray(img[:,:,:3]) # convert to grayscale using only the first 3 channels (RGB)
    print("Image converted to Grayscale!")
elif img.ndim == 3 and img.shape[2] == 3:
    print("Image is RGB!")
else:
    print("Image is Grayscale!")

# Load pre-trained StarDist model
model = StarDist2D.from_pretrained("2D_versatile_fluo")

# Normalize image
img_norm = normalize(img, 1, 99.8)

# Predict segmentation
labels, _ = model.predict_instances(img_norm)

# Visualization
lbl_cmap = random_label_cmap()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')

axes[1].imshow(img, cmap='gray')
axes[1].imshow(labels, cmap=lbl_cmap, alpha=0.5)
axes[1].set_title('StarDist Segmentation Overlay')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()



# Load image
img = imread("sample_data/Image 01_denoise.tiff")

# Convert image to grayscale if it has more than 3 channels or is RGBA
if img.ndim == 3 and (img.shape[2] > 3 or img.shape[2] == 4): #check if image has more than 3 channels or is RGBA
    img = rgb2gray(img[:,:,:3]) # convert to grayscale using only the first 3 channels (RGB)
    print("Image converted to Grayscale!")
elif img.ndim == 3 and img.shape[2] == 3:
    print("Image is RGB!")
else:
    print("Image is Grayscale!")

# Load pre-trained StarDist model
model = StarDist2D.from_pretrained("2D_versatile_fluo")

# Normalize image
img_norm = normalize(img, 1, 99.8)

# Predict segmentation
labels, _ = model.predict_instances(img_norm)

# Visualization
lbl_cmap = random_label_cmap()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Denoised Imaged')

axes[1].imshow(img, cmap='gray')
axes[1].imshow(labels, cmap=lbl_cmap, alpha=0.5)
axes[1].set_title('StarDist Segmentation Overlay')

for ax in axes:
    ax.axis('off')

plt.tight_layout()

