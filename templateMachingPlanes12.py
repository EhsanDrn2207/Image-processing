import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# File paths
image_path = "pictures/all_plane.jpg"
template_path = "pictures/Screenshot_plane.png"

# Reading the image and template in grayscale
image = cv.imread(image_path, 0)
template = cv.imread(template_path, 0)

# Get the dimensions of the template
w, h = template.shape[::-1]

# Perform template matching
result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

# Define a threshold for detecting matches
threshold = 0.5
locations = np.where(result >= threshold)

# Draw rectangles around the matches
for point in zip(*locations[::-1]):
    cv.rectangle(image, point, (point[0] + w, point[1] + h), (255, 255, 0), 2)

# Display the result
plt.imshow(image, cmap='gray')
plt.title('Template Matching Result')
plt.axis('off')
plt.show()
