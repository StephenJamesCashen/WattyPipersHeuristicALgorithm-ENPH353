import numpy as np
import cv2


def undistort(img, path="Competition\\Data_Collection\\Camera_Calibration"):
    mtx = np.load(path + "\\mtx.npy")
    dist = np.load(path + "\\dist.npy")

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
