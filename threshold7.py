import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

# File path
file_path = "pictures/brain.jpg"

# Reading and converting the image
img = cv.imread(file_path, cv.IMREAD_COLOR)
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Applying different thresholding techniques
ret, threshold = cv.threshold(imgray, thresh=50, maxval=255, type=cv.THRESH_BINARY)
ret2, threshold2 = cv.threshold(imgray, thresh=100, maxval=255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)
threshold3 = cv.adaptiveThreshold(imgray, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv.THRESH_BINARY, blockSize=115, C=1)

# Setup plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

# Configuration and displaying images
axes[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axes[0, 0].axis("off")
axes[0, 0].title.set_text('Original Image')

axes[0, 1].imshow(threshold, cmap='gray')
axes[0, 1].axis("off")
axes[0, 1].title.set_text('Binary Threshold')

axes[1, 0].imshow(threshold2, cmap='gray')
axes[1, 0].axis("off")
axes[1, 0].title.set_text('Otsu Threshold')

axes[1, 1].imshow(threshold3, cmap='gray')
axes[1, 1].axis("off")
axes[1, 1].title.set_text('Adaptive Gaussian Threshold')

# Show plot
plt.show()
