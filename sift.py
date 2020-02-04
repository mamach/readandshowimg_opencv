# import numpy as np
# works onlcy with 3.4 version of opencv
import cv2 as cv

img = cv.imread('./images/template.jpg', cv.IMREAD_COLOR)          # queryImage
img2 = cv.imread('./images/matching_template1.jpg', cv.IMREAD_COLOR) # trainImage



# img = cv.imread('home.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)


cv.startWindowThread()
cv.namedWindow("preview")
cv.imshow("preview", img)
cv.waitKey()

