import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# File path
image_path = "path/to/your/image.jpg"

# Reading the image in color
image = cv.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# Colors list for plotting
colors = ['b', 'g', 'r']

# Plotting histograms for each color channel
plt.figure(figsize=(12, 6))
for i, color in enumerate(colors):
    histogram = cv.calcHist(images=[image], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(histogram, color=color, label=f'{color.upper()} channel')

# Enhancing the plot
plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()
