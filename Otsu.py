import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from PIL import Image

# Load the image
gray_image2 = imread('sample_data/Image 01_denoise.tiff')

# Drop alpha channel if present (keep only RGB for grayscale conversion)
if gray_image2.ndim == 3 and gray_image2.shape[2] == 4:
    gray_image2 = gray_image2[:, :, :3]

# Convert to grayscale
if gray_image2.ndim == 3:
    gray_image2 = rgb2gray(gray_image2)
    gray_image2 = (gray_image2 * 255).astype(np.uint8)

# Compute histogram and normalized probabilities
hist, bins = np.histogram(gray_image2.flatten(), bins=256, range=[0, 256])
hist_normalized = hist / hist.sum()
bins_center = (bins[:-1] + bins[1:]) / 2
mu_T = np.sum(bins_center * hist_normalized)

# Compute between-class variance for each threshold
variances = np.zeros(256)
for k in range(255):
    w0 = np.sum(hist_normalized[:k+1])
    w1 = 1 - w0
    if w0 <= 0 or w1 <= 0:
        continue
    mu_0 = np.sum(bins_center[:k+1] * hist_normalized[:k+1]) / w0
    mu_1 = np.sum(bins_center[k+1:] * hist_normalized[k+1:]) / w1
    variances[k] = w0 * w1 * (mu_0 - mu_1) ** 2

# Find the optimal threshold
optimal_threshold = np.argmax(variances)
print(f"Optimal Threshold (Otsu's method): {optimal_threshold}")

# Create binary image
binary_image = (gray_image2 > optimal_threshold).astype(np.uint8) * 255

# Save binary result as PNG
Image.fromarray(binary_image).save('sample_data/Otsu_result.png')

# Display the binary image
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image (Thresholded)')
plt.axis('off')
plt.show()

# Display histogram with threshold line
plt.bar(bins_center, hist, align='center')
plt.axvline(optimal_threshold, color='r', label=f'Threshold = {optimal_threshold}')
plt.legend()
plt.title('Histogram with Otsu Threshold')
plt.show()
