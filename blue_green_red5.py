import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# File path
file_path = "pictures/blobs2.png"

# Reading the image
img = cv.imread(file_path, cv.IMREAD_COLOR)

# Splitting the color channels
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]

# Create blank images with zeros
zeros = np.zeros_like(b)

# Merge channels to create colored representations
blue_img = cv.merge([b, zeros, zeros])
green_img = cv.merge([zeros, g, zeros])
red_img = cv.merge([zeros, zeros, r])

# Setup plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(20, 10))

# Configuration and displaying images
# ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# ax1.axis("off")
# ax1.title.set_text('Original Image')

# ax2.imshow(cv.cvtColor(blue_img, cv.COLOR_BGR2RGB))
# ax2.axis("off")
# ax2.title.set_text('Blue Channel')

# ax3.imshow(cv.cvtColor(green_img, cv.COLOR_BGR2RGB))
# ax3.axis("off")
# ax3.title.set_text('Green Channel')

# ax4.imshow(cv.cvtColor(red_img, cv.COLOR_BGR2RGB))
# ax4.axis("off")
# ax4.title.set_text('Red Channel')

# Show plot
# plt.show()


# Configuration and displaying images
ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax1.axis("off")
ax1.title.set_text('Original Image')

ax2.imshow(b, cmap='gray')
ax2.axis("off")
ax2.title.set_text('Blue')

ax3.imshow(g, cmap='gray')
ax3.axis("off")
ax3.title.set_text('Green')

ax4.imshow(r, cmap='gray')
ax4.axis("off")
ax4.title.set_text('Red')

#  Show plot
plt.show()
