import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# File path
image_path = "path/to/your/image.jpg"

# Reading the image in grayscale
image = cv.imread(image_path, flags=0)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Calculating the original histogram
img_hist = cv.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# Histogram Equalization
equalized_histogram = cv.equalizeHist(src=image)
img_equal_hist = cv.calcHist(images=[equalized_histogram], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(image)
cl_equal_hist = cv.calcHist(images=[cl1], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# Plotting
plt.figure(figsize=(18, 6))

# Original Image and Histogram
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.plot(img_hist, color='black')
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

# Equalized Image and Histogram
plt.subplot(2, 3, 3)
plt.imshow(equalized_histogram, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.plot(img_equal_hist, color='black')
plt.title('Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

# CLAHE Image and Histogram
plt.subplot(2, 3, 5)
plt.imshow(cl1, cmap='gray')
plt.title('CLAHE Image')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.plot(cl_equal_hist, color='black')
plt.title('CLAHE Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

# Display the plots
plt.tight_layout()
plt.show()
