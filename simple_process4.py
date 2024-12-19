import cv2 as cv
import matplotlib.pyplot as plt

# File path
file_path = "pictures/shapes.png"

# Reading and converting the image
img = cv.imread(file_path, cv.IMREAD_COLOR)
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Make a black hole in the middle of the picture
img[100:150, 100:150] = [0, 0, 0]

# White and black images have only 1 color channel
imgray[100:300, 200:300] = 255

# Selecting a region of pixels from the grayscale image
img2 = imgray[100:150, 10:150]

# Setup plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Configuration and displaying the images
ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax1.axis("off")
ax1.title.set_text('Original Image with Black Hole')

ax2.imshow(imgray, cmap='gray')
ax2.axis("off")
ax2.title.set_text('Gray Image with White Region')

ax3.imshow(img2, cmap='gray')
ax3.axis("off")
ax3.title.set_text('Cropped Region from Gray Image')

# Show plot
plt.show()


# x_blob = img[250:370, 50:180]
# img[50:170, 350:480] = x_blob


# img.shape
# imgray.shape
# img.size
# img.dtype

