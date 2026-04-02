import cv2
import numpy as np


img = cv2.imread("variant6.png")

height, width = img.shape[:2]

new_height = height * 2
new_width = width * 2
new_img = cv2.resize(img, (new_width, new_height))

cv2.imshow("image", img)
cv2.imshow("new image", new_img)

cv2.waitKey(0)
cv2.destroyAllWindows()