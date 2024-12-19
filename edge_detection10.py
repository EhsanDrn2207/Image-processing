import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# File path
file_path = "pictures/H.jpg"

# Reading and preprocessing the image
img = cv.imread(file_path, flags=0)
image_noise_removed = cv.GaussianBlur(src=img, ksize=(7, 7), sigmaX=0)

# Edge detection techniques
laplacian = cv.Laplacian(image_noise_removed, cv.CV_64F)
sobelx = cv.Sobel(image_noise_removed, cv.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv.Sobel(image_noise_removed, cv.CV_64F, 0, 1, ksize=5)
canny = cv.Canny(image_noise_removed, threshold1=100, threshold2=200)

# Setup plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Configuration and displaying images
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].axis("off")
axes[0, 0].title.set_text('Original')

axes[0, 1].imshow(image_noise_removed, cmap='gray')
axes[0, 1].axis("off")
axes[0, 1].title.set_text('Noise Removed')

axes[0, 2].imshow(laplacian, cmap='gray')
axes[0, 2].axis("off")
axes[0, 2].title.set_text('Laplacian')

axes[1, 0].imshow(sobelx, cmap='gray')
axes[1, 0].axis("off")
axes[1, 0].title.set_text('Sobel X')

axes[1, 1].imshow(sobely, cmap='gray')
axes[1, 1].axis("off")
axes[1, 1].title.set_text('Sobel Y')

axes[1, 2].imshow(canny, cmap='gray')
axes[1, 2].axis("off")
axes[1, 2].title.set_text('Canny')

# Show plot
plt.tight_layout()
plt.show()
