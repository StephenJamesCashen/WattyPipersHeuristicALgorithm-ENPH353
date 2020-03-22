import numpy as np
import cv2
import glob

nrows = 4
ncolumns = 11

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

PATH = 'Competition\\Data_Collection\\Camera_Calibration\\'

objp = []
grid_size = 0.03  # 3cm, or whatever

for i in range(ncolumns):
    for j in range(nrows):
        objp.append((i * grid_size, (2 * j + i % 2) * grid_size, 0))

objp = np.array(objp).astype('float32')

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('Competition\\Data_Collection\\Camera_Calibration\\image_*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, circles = cv2.findCirclesGrid(gray, (nrows, ncolumns), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    # If found, add object point
    # s, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        imgpoints.append(circles)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nrows, ncolumns), circles, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.save("Competition\\Data_Collection\\Camera_Calibration\\mtx", mtx)
np.save("Competition\\Data_Collection\\Camera_Calibration\\dist", dist)

img = cv2.imread('Competition\\Data_Collection\\Camera_Calibration\\test_image.png')

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('Competition\Data_Collection\Camera_Calibration\\calibresult.png', dst)
