import cv2 as cv
import matplotlib.pyplot as plt

# File path
file_path = "pictures/Artur Morgan.jpg"

# Reading and converting images
img = cv.imread(file_path, cv.IMREAD_UNCHANGED)
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Drawing shapes on the grayscale image
cv.line(img=imgray, pt1=(0, 200), pt2=(200, 0), color=(0, 0, 0), thickness=15)
cv.rectangle(img=imgray, pt1=(5, 5), pt2=(1325, 1150), color=(0, 0, 0), thickness=12)

# Setup plot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

# Configuration and displaying images
ax1.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax1.axis("off")
ax1.title.set_text('Original Image')

ax2.imshow(imgray, cmap='gray')
ax2.axis("off")
ax2.title.set_text('Gray Image')

# Show plot
plt.show()


# cv.circle(img=img, center=(200, 150), radius=50, color=(0, 255, 255), thickness=-1)
# points = np.array([[30, 50], [300, 200], [100, 70], [40, 100]], dtype=np.int32)
# cv.polylines(img, [points], True, (255, 0, 0), 5)


# cv.imshow(winname='Original', mat=img)
# cv.imshow(winname='Gray', mat=imgray)

# cv.waitKey(0)
# cv.destroyAllWindows()
