import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# File path
image_path = "path/to/your/image.jpg"

# Reading the image in color
image = cv.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Creating a mask for the top-left quadrant
w, h, c = image.shape
mask = np.zeros(shape=image.shape[0:2], dtype=np.uint8)
mask[0:int(w/2), 0:int(h/2)] = 255

# Plotting histograms for each color channel
plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r']
for i, color in enumerate(colors):
    histogram = cv.calcHist(images=[image], channels=[i], mask=mask, histSize=[256], ranges=[0, 256])
    plt.plot(histogram, color=color, label=f'{color.upper()} channel')

# Enhancing the plot
plt.title('Color Histogram with Mask')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
