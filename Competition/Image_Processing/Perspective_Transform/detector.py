import cv2
import numpy as np
import Competition.data_collection_utils as dcu
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

img = cv2.imread("ref_img.png")
img = cv2.resize(img, (665, 915))

print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

keypoints = detector_ref.detect(gray)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]),
                                      (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("blobs1.jpg", im_with_keypoints)

img = cv2.imread("image_1.png")
img = dcu.undistort(img)
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

keypoints = detector1.detect(gray)

centers = np.array([[kp.pt] for kp in keypoints], dtype=np.float32)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]),
                                      (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.drawChessboardCorners(img, (4,11), centers, True)

cv2.imwrite("blobs2.jpg", im_with_keypoints)


