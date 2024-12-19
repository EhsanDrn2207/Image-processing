import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# File paths
image1_path = "path/to/book1.jpg"
image2_path = "path/to/book2.jpg"

# Reading the images
image1 = cv.imread(image1_path)
image2 = cv.imread(image2_path)
if image1 is None or image2 is None:
    raise FileNotFoundError("One of the images was not found.")

# ORB Feature Detector
feat_orb = cv.ORB_create(nfeatures=1000)

# Detecting keypoints and descriptors
orb_keypoints1, orb_descriptors1 = feat_orb.detectAndCompute(image1, None)
orb_keypoints2, orb_descriptors2 = feat_orb.detectAndCompute(image2, None)

# Brute-Force Matcher with Hamming distance
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Matching descriptors
matches = bf.match(queryDescriptors=orb_descriptors1, trainDescriptors=orb_descriptors2)

# Sorting matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Drawing the matches
image_matches = cv.drawMatches(img1=image1, keypoints1=orb_keypoints1, img2=image2, keypoints2=orb_keypoints2, matches1to2=matches[:100], outImg=None, flags=2)

# Displaying the matches
plt.figure(figsize=(20, 10))
plt.imshow(cv.cvtColor(image_matches, cv.COLOR_BGR2RGB))
plt.title('ORB Feature Matching')
plt.axis('off')
plt.show()
