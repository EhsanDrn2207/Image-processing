import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# File path
file_path = "pictures/H.jpg"

# Reading the image
image = cv.imread(file_path)
image_gr = cv.imread(file_path, 0)

# Harris Corner Detection
corners = cv.cornerHarris(image_gr, blockSize=10, ksize=21, k=0.08)
corner_dilated = cv.dilate(corners, None)

# Marking the corners in red
image[corner_dilated > 0.01 * corner_dilated.max()] = [0, 0, 255]

# Displaying the image
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title('Harris Corners')
plt.axis('off')
plt.show()
