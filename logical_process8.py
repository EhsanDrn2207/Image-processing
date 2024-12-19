import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# File paths
file_path1 = "pictures/Nature.jpg"
file_path2 = "pictures/Dog.jpg"

# Reading images
img1 = cv.imread(file_path1, cv.IMREAD_COLOR)
img2 = cv.imread(file_path2, cv.IMREAD_COLOR)

# Check if images are loaded correctly
if img1 is None or img2 is None:
    raise ValueError("One or both images could not be loaded. Check the file paths.")

# Ensure both images have the same size
if img1.shape != img2.shape:
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

# Convert images to grayscale
img1gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Create masks
ret, mask_img = cv.threshold(img2gray, thresh=40, maxval=255, type=cv.THRESH_BINARY)
mask_invert = cv.bitwise_not(mask_img)

# Apply masks
img1_mask = cv.bitwise_and(img1, img1, mask=mask_invert)
img2_mask = cv.bitwise_and(img2, img2, mask=mask_img)

# Combine masked images
imgadded = cv.add(img1_mask, img2_mask)

# Setup plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

# Configuration and displaying images
axes[0, 0].imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
axes[0, 0].axis("off")
axes[0, 0].title.set_text('Image 1')

axes[0, 1].imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
axes[0, 1].axis("off")
axes[0, 1].title.set_text('Image 2')

axes[0, 2].imshow(mask_img, cmap='gray')
axes[0, 2].axis("off")
axes[0, 2].title.set_text('Mask Image')

axes[1, 0].imshow(mask_invert, cmap='gray')
axes[1, 0].axis("off")
axes[1, 0].title.set_text('Inverted Mask')

axes[1, 1].imshow(cv.cvtColor(imgadded, cv.COLOR_BGR2RGB))
axes[1, 1].axis("off")
axes[1, 1].title.set_text('Added Image')

axes[1, 2].imshow(cv.cvtColor(img1_mask, cv.COLOR_BGR2RGB))
axes[1, 2].axis("off")
axes[1, 2].title.set_text('Image 1 with Mask')

# Show plot
plt.show()
