import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# File path
image_path = "path/to/your/image.jpg"

# Reading the image in original format
image = cv.imread(image_path, flags=cv.IMREAD_UNCHANGED)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

# SIFT
feat_sift = cv.xfeatures2d.SIFT_create()

# SURF
feat_surf = cv.xfeatures2d.SURF_create()

# ORB
feat_orb = cv.ORB_create(nfeatures=100)

# Detecting keypoints and descriptors
sift_keypoints, sift_descriptors = feat_sift.detectAndCompute(image, None)
surf_keypoints, surf_descriptors = feat_surf.detectAndCompute(image, None)
orb_keypoints, orb_descriptors = feat_orb.detectAndCompute(image, None)

# Drawing keypoints
image_sift = cv.drawKeypoints(image=image, keypoints=sift_keypoints, outImage=None)
image_surf = cv.drawKeypoints(image=image, keypoints=surf_keypoints, outImage=None)
image_orb = cv.drawKeypoints(image=image, keypoints=orb_keypoints, outImage=None)

# Plotting
plt.figure(figsize=(18, 6))

# SIFT Keypoints
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image_sift, cv.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.axis('off')

# SURF Keypoints
plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(image_surf, cv.COLOR_BGR2RGB))
plt.title('SURF Keypoints')
plt.axis('off')

# ORB Keypoints
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(image_orb, cv.COLOR_BGR2RGB))
plt.title('ORB Keypoints')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
