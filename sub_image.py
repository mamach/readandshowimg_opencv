# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
import pdb

# img1 = cv2.imread('./images/template.jpg', 0)          # queryImage
# img2 = cv2.imread('./images/matching_template2.jpg', 0) # trainImage
import matplotlib.pyplot as plt
import cv2  
import numpy as np  
image = cv2.imread("./images/template.jpg",0)  
template = cv2.imread("./images/snip2.jpg",0)
# template = cv2.imread("./images/matching_template3.jpg")

height, width = template.shape




result = cv2.matchTemplate(image, template,cv2.TM_CCOEFF_NORMED)  
origin = np.unravel_index(result.argmax(),result.shape)
# pdb.set_trace()
print(np.unravel_index(result.argmax(),result.shape))

roi = image[origin[0]:origin[0]+height, origin[1]:origin[1]+width]
# roi = image[530:(530+width), 130:(130+height)]


print(origin[0])
print(origin[0] + width)
print(origin[1])
print(origin[1] + height)

cv2.imwrite("roi.png", roi)

# cv2.startWindowThread()
# cv2.namedWindow("preview")
# cv2.imshow("preview", image)
# cv2.waitKey()

plt.imshow(roi)
plt.show()
