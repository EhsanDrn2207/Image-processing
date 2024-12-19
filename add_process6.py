import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# File paths
file_path1 = "pictures/OIP.jpg"
file_path2 = "pictures/OIP2.jpg"

# Reading images
img1 = cv.imread(file_path1, cv.IMREAD_COLOR)
img2 = cv.imread(file_path2, cv.IMREAD_COLOR)

# Check if images are loaded correctly
if img1 is None or img2 is None:
    raise ValueError("One or both images could not be loaded. Check the file paths.")

# Ensure both images have the same size
if img1.shape != img2.shape:
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

# Element-wise addition (may cause overflow)
added_elementwise = img1 + img2

# Saturated addition using OpenCV
added_saturated = cv.add(img1, img2)

# Weighted addition
alpha = 0.8
beta = 1 - alpha
gamma = 0
added_weighted = cv.addWeighted(img1, alpha, img2, beta, gamma)

# Setup plot
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

# Configuration and displaying images
axes[0].imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
axes[0].axis("off")
axes[0].title.set_text('Image 1')

axes[1].imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
axes[1].axis("off")
axes[1].title.set_text('Image 2')

axes[2].imshow(cv.cvtColor(added_elementwise, cv.COLOR_BGR2RGB))
axes[2].axis("off")
axes[2].title.set_text('Element-wise Addition')

axes[3].imshow(cv.cvtColor(added_weighted, cv.COLOR_BGR2RGB))
axes[3].axis("off")
axes[3].title.set_text('Weighted Addition')

# Show plot
plt.show()
