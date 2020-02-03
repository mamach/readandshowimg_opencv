import numpy as np
import cv2
import matplotlib.pyplot as plt
# %matplotlib inline # if you are running this code in jupyter notebook

img = cv2.imread('./images/logo.jpeg') # reads image 'opencv-logo.png' as grayscale
# cv2.imshow("OpenCV Image Reading", img)
# height, width, channels = img.shape

print (img.shape)

# plt.imshow(img, cmap='gray')
# cv2.waitKey()
# print(cv2.__version__)

cv2.startWindowThread()
cv2.namedWindow("preview")
cv2.imshow("preview", img)
cv2.waitKey()