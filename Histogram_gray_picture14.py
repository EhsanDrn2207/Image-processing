import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# File path
image_path = "path/to/your/image.jpg"

# Reading the image in grayscale
image = cv.imread(image_path, flags=0)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Calculating the histogram
histogram = cv.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# Plotting
plt.figure(figsize=(12, 6))

# Displaying the image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Plotting the histogram
plt.subplot(1, 2, 2)
plt.plot(histogram, color='black')
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

# Showing the plot
plt.tight_layout()
plt.show()
