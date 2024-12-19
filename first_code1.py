import cv2 as cv
import matplotlib.pyplot as plt

# File path
file_path = "pictures/butterfly2.jpg"

# Reading images
img = cv.imread(file_path)
img_gray = cv.imread(file_path, flags=cv.IMREAD_GRAYSCALE)
imgcolor = cv.imread(file_path, flags=cv.IMREAD_COLOR)

# Setup plot
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 20))

# Configuration and displaying images
ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax1.axis("off")
ax1.title.set_text('Original Image')

ax2.imshow(cv.cvtColor(imgcolor, cv.COLOR_BGR2RGB))
ax2.axis("off")
ax2.title.set_text('Color Image')

ax3.imshow(img_gray, cmap='gray')
ax3.axis("off")
ax3.title.set_text('Gray Image')

# Show plot
plt.show()

# Save image (optional)
# cv.imwrite(filename='white_and_blank.jpg', img=img)
