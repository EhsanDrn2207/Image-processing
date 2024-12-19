import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# File paths
image_path = "pictures/all_glasses.jpg"
template_path = "pictures/Screenshot_glass.png"

# Reading the image and template in grayscale
image = cv.imread(image_path, 0)
template = cv.imread(template_path, 0)

# Verify dimensions
print("Image dimensions:", image.shape)
print("Template dimensions:", template.shape)

# Resize the template if necessary
if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
    template = cv.resize(template, (image.shape[1], image.shape[0]))

# Get the dimensions of the template
w, h = template.shape[::-1]

# Perform template matching
result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

# Define a threshold for detecting matches
threshold = 0.5
locations = np.where(result >= threshold)

# Draw rectangles around the matches
for point in zip(*locations[::-1]):
    cv.rectangle(image, point, (point[0] + w, point[1] + h), (0, 0, 0), 2)

# Display the result
plt.imshow(image, cmap='gray')
plt.title('Template Matching Result')
plt.axis('off')
plt.show()
