import cv2 as cv
import matplotlib.pyplot as plt

# File path
file_path = "pictures/butterfly2.jpg"

# Reading the image
img = cv.imread(file_path, cv.IMREAD_COLOR)

# Adding text to the image
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, text="Hello World!", org=(180, 35), fontFace=font, fontScale=1, color=(255, 255, 255), thickness=3, lineType=cv.LINE_AA)

# Setup plot
_, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 20))

# Configuration and displaying the image
ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax1.axis("off")
ax1.title.set_text('Image with Text')

# Show plot
plt.show()
