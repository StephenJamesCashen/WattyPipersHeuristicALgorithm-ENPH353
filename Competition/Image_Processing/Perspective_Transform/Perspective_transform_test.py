# import the necessary packages
import numpy as np
import cv2
import Competition.data_collection_utils as dcu
from Competition.global_variables import path
params_1 = cv2.SimpleBlobDetector_Params()
params_1.filterByArea = True
params_1.filterByCircularity = False
params_1.filterByConvexity = True
params_1.filterByInertia = False
params_1.filterByColor = False

params_1.minArea = 7
params_1.maxArea = 1000

params_1.minThreshold = 125
params_1.maxThreshold = 255
params_1.minConvexity = 0.9
params_1.maxConvexity = 1

params_1.minRepeatability = 4

params_1.minDistBetweenBlobs = 1

params_ref = cv2.SimpleBlobDetector_Params()
params_ref.filterByArea = True
params_ref.filterByCircularity = True
params_ref.filterByConvexity = True
params_ref.filterByInertia = False
params_ref.filterByColor = False

params_ref.minArea = 500
params_ref.maxArea = 50000

params_ref.minThreshold = 150
params_ref.maxThreshold = 255
params_ref.minConvexity = 0.9
params_ref.maxConvexity = 1
#
params_ref.minCircularity = 0.8
params_ref.maxCircularity = 1

detector1 = cv2.SimpleBlobDetector_create(params_1)
detector_ref = cv2.SimpleBlobDetector_create(params_ref)


nrows = 4
ncolumns = 11

objp = []
grid_size = 0.03  # 3cm, or whatever

for i in range(ncolumns):
    for j in range(nrows):
        objp.append((i * grid_size, (2 * j + i % 2) * grid_size, 0))

objp = np.array(objp).astype('float32')

# Arrays to store object points and image points from all the images.
ref_image = cv2.imread("ref_img.png")

img_1 = cv2.imread("image_1.png")

img_1 = dcu.undistort(img_1)

gray_img = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret_2, circles = cv2.findCirclesGrid(gray_img, (nrows, ncolumns), flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                                     blobDetector=detector1)

ret_1, ref_circles = cv2.findCirclesGrid(ref_image, (nrows, ncolumns), flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                                         blobDetector=detector_ref)


objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

objpoints.append(objp)

imgpoints.append(circles)

# # Draw and display the corners
# drawn_image = cv2.drawChessboardCorners(img_1, (nrows, ncolumns), circles, ret_1)
# drawn_reference = cv2.drawChessboardCorners(ref_image, (nrows, ncolumns), ref_circles, ret_1)
#
# cv2.imshow('img', drawn_reference)
# cv2.waitKey(0)
# cv2.imshow('img', drawn_image)
# cv2.waitKey(0)

h, mask = cv2.findHomography(circles, ref_circles)
np.save("homography.npy", h)
np.save("shape.npy", ref_image.shape)
#
# cal_out = cv2.warpPerspective(img_1, h, (ref_image.shape[1], ref_image.shape[0]))
# cv2.imwrite("cal_out.png", cal_out)
#
# test_image = cv2.imread("test_image.png")
#
# gray_test = cv2.cv
#
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# test_out = cv2.warpPerspective(test_image, h, (ref_image.shape[1], ref_image.shape[0]))
#
#
#
# cv2.imwrite("test_out.png", test_out)
