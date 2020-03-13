# import the necessary packages
import numpy as np
import cv2
import time

x, y = 400, 800

img1_square_corners = np.float32([(254,328), (344,324), (268,282),(331,282)])
img2_quad_corners = np.float32([(x,y+80),(x+60,y+80),(x,y),(x+60,y)])

h, mask = cv2.findHomography(img1_square_corners, img2_quad_corners)
cal_in = cv2.imread("Competition\\Image_Processing\\calibration_image.png")

cal_out = cv2.warpPerspective(cal_in, h, (3000,3000))

test_in = cv2.imread("Competition\\Image_Processing\\test_image.png")

test_out = cv2.warpPerspective(test_in, h, (3000,3000))

cv2.imshow("output", test_out) 
cv2.waitKey(0)
