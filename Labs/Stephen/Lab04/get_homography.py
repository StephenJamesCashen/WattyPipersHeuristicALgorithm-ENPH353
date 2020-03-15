import numpy as np
import cv2


img = cv2.imread("block_pattern.jpg", cv2.IMREAD_GRAYSCALE)  # get reference image and convert to grey scale


cap = cv2.VideoCapture(0) # Captures from device 0. 


# Features
sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)

# Setting up Flann Algorithm
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Get the image
_, frame = cap.read()
# Convert to grey scale
grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Run sift algorithm
kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

# Match features with the flann algorithm
matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

# Filter out bad points
good_points = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_points.append(m)

# Seperate points into
query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

# Given set of points and matching points find the perspective transform matrix and a mask defining the region of interest.
matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()

# Get the corner points of the reference image
h, w = img.shape
pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

# Send the corner points to the corner of the object
dst = cv2.perspectiveTransform(pts, matrix)

# Draw a box!
homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
cv2.imshow("Homography", homography)


# Pseudo code

def get_centroid(corners):
    centroid = [0,0]
    for corner in corner:
        centroid += corner/4

    return centroid

center/2 = frame.shape

centroid = get_centroid(dst):

def get_distance_metric(matrix)
    return determinant(matrix)**-(1/2)

x_error = center[0]-centroid[0] # Calculate 

distance = get_distance_metric(matrix)

distance_calibration = 10 # Need to calibrate this

d_error = center[0]-distance_calibration[0]
