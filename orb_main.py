# reference: https://pythonprogramming.net/feature-matching-homography-python-opencv-tutorial/

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb

img1 = cv2.imread('./images/template.jpg', 0)          # queryImage
img2 = cv2.imread('./images/matching_template2.jpg', 0) # trainImage

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
matchIndex = 0

# get matches less than 10 here
for match in matches:
    if match.distance <= 10.0:
        matchIndex += 1
    else:
        break
# return the matchIndex count here. Higher the count better the match.
print( matchIndex)

# pdb.set_trace()

# show the image here
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],None, flags=2)
plt.imshow(img3)
plt.show()
