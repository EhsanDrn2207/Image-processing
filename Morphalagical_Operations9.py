import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


original_img = cv.imread(filename="", flags=0)

_, mask = cv.threshold(original_img, 25, 255, cv.THRESH_BINARY)

# Erosion
kernel1 = np.ones((5,5), np.unint8)
# eroded_img = cv.erode(mask, kernel=kernal, iterations=1)


# Dilation
kernel2 = np.ones((3,3), np.uint8)
# dilated_img = cv.dilate(mask, kernel=kernel, iterations=1)
# kernal = np.ones((5,5), np.unint8)
# eroded_img = cv.erode(dilated_img, kernel=kernal, iterations=1)


# Closing
# closed_img1 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel=kernel1)
# closed_img2 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel=kernel2)


# Opening
opened_img1 = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=kernel1)
opened_img2 = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=kernel2)


# Gradient
gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernel=kernel1)
gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernel=kernel2)



fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 20))
cmap_val = 'gray'
ax1.axis("off")
ax1.title.set_text('original mask')
ax2.axis("off")
ax2.title.set_text('eroded_img')
ax3.axis("off")
ax3.title.set_text('eroded_img2')

ax1.imshow(mask, cmap=cmap_val)
ax2.imshow(opened_img1, cmap=cmap_val)
ax3.imshow(opened_img2, cmap=cmap_val)

plt.show()
