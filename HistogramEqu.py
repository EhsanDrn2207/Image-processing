import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# image = cv.imread(filename="", flags=0)
# histogram = cv.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# plt.subplot(121); plt.imshow(image, 'gray')
# plt.subplot(122); plt.plot(histogram)
# plt.show()



# image = cv.imread("")
# colors = ['b', 'g', 'r']
# for i, color in enumerate(colors):
#     histogram = cv.calcHist(images=[image], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
#     plt.plot(histogram, color = color)
# plt.show()



# image = cv.imread("")
# w,h,c = image.shape
# mask = np.zeros(shape=image.shape[0:2], dtype=np.uint8)
# mask[0:int(w/2), 0:int(h/2)] = 255

# colors = ['b', 'g', 'r']
# for i, color in enumerate(colors):
#     histogram = cv.calcHist(images=[image], channels=[i], mask=None, histSize=[256], ranges=[0, 256])
#     plt.plot(histogram, color = color)
# plt.show()



# image = cv.imread(filename="", flags=0)
# img_hist = cv.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
# equalized_histogram = cv.equalizeHist(src=image)
# img_equal_hist = cv.calcHist(images=[equalized_histogram], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(image)
# cl_equal_hist = cv.calcHist(images=[cl1], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
