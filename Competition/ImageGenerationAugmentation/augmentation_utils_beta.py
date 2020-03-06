#!/usr/bin/env python
import cv2
import numpy as np

from skimage.transform import rotate as scikit_rotate
from skimage.transform import AffineTransform, warp
from skimage.util import random_noise
from random import seed, randint

HSB_MAX = 255
PERSPECTIVE_BORDER = 7
seed(1)

"""
Image augmentation methods to get more plate images
Most code was sourced / modified from:
https://github.com/govinda007/Images/blob/master/augmentation.ipynb

Credit to original author (if another resource was used, it's also specified)

This is a not object oriented mod of the original augmentation_utils.py
"""

# ---------- General helpers, implemented by moi -------


def new_hsb(old_val, offset):
    """
    Helper function for hsb, to cap off vals within acceptable range
    """
    new_val = old_val + offset
    if (new_val < 0):
        new_val = 0
    elif (new_val > HSB_MAX):
        new_val = HSB_MAX
    return new_val


def generate_corners(img):
    """
    Helper function to generate our 4 points for perspective transform
    Important: points are generated in a consistent order! I'll do:
    1. top-left
    2. top-right
    3. bottom-right
    4. bottom-left
    """
    width = img.shape[1]
    height = img.shape[0]

    top_left = (randint(0, width // PERSPECTIVE_BORDER),
                randint(0, height // PERSPECTIVE_BORDER))

    top_right = (randint((PERSPECTIVE_BORDER - 1)
                 * width // PERSPECTIVE_BORDER, width - 1),
                 randint(0, height // PERSPECTIVE_BORDER))

    bottom_right = (randint((PERSPECTIVE_BORDER - 1)
                    * width // PERSPECTIVE_BORDER, width - 1),
                    randint((PERSPECTIVE_BORDER - 1)
                    * height // PERSPECTIVE_BORDER, height - 1))

    bottom_left = (randint(0, width // PERSPECTIVE_BORDER),
                   randint((PERSPECTIVE_BORDER - 1)
                   * height // PERSPECTIVE_BORDER, height - 1))

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = top_left
    rect[1] = top_right
    rect[2] = bottom_right
    rect[3] = bottom_left

    return rect

# --------- Perspective transform helpers ----------------------
# Code modified from: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/


def four_point_transform(img):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = generate_corners(img)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

# ------------ API ------------------------------


def scale(img, c_x, c_y):
    """
    Scales image
    Params: c_x - horizontal scaling factor (positive int)
            c_y - vertical scaling factor (positive int)
    Returns: scaled image
    """
    return cv2.resize(img, (int(img.shape[1] * c_x),
                            int(img.shape[0] * c_y)))


def rotate(img, phi):
    """
    Rotates image
    Params: phi - angle of rotation (degrees)
    Returns: rotated image
    """
    return scikit_rotate(img, angle=phi)


def blur(img, kernel_size):
    """
    Adds uniform blur
    Params: img, kernel_size
    Returns: blurred image
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def motion_blur(img, kernel_size):
    """
    Adds horizontal motion blur

    Params: img, kernel_size
    Returns: motion blurred image
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    # normalise
    kernel /= kernel_size

    # Apply horizontal kernel
    return cv2.filter2D(img, -1, kernel)


def noise(img, param):
    """
    Add noise
    Params: dunno yet
    Returns: noisier image
    """
    return random_noise(img, mode=param, seed=randint(0, 100))


def rand_noise(img):
    """
    Randomly picks from the nice modes
    """
    modes = ['gaussian', 'localvar', 'poisson', 'speckle']
    return noise(img, modes[randint(0, 3)])


def hsb(img, hue_on=False):
    """
    Change hue, saturation, brightness
    If any resulting pixels have invalid hsb, 
    the values are capped at 255 and 0
    Params: hue - fractional change of hue
            saturation - fractional change of saturation
    Returns: HSB modified image
    """
    hsb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.waitKey(2000)

    # randomly generate offsets. Randint used 3 times so that it clusters towards middle values
    h = 0
    if (hue_on):
        h = randint(-50, 50) + randint(-50, 50) + randint(-50, 50)
        s = randint(-50, 50) + randint(-50, 50) + randint(-50, 50)
        b = randint(-50, 50) + randint(-50, 50) + randint(-50, 50)

        for i in range(hsb.shape[0]):
            for j in range(hsb.shape[1]):
                hsb[i][j][0] = new_hsb(hsb[i][j][0], h)
                hsb[i][j][1] = new_hsb(hsb[i][j][1], s)
                hsb[i][j][2] = new_hsb(hsb[i][j][2], b)
        
        # note that you get fun fun colours if you return img as hsv!
        return cv2.cvtColor(hsb, cv2.COLOR_HSV2BGR)


def shift(img, dx, dy):
    """
    Shifts image
    Params: dx - shift of the image in pixels in x
            dy - shift of image in pixels in y
                    < 0 if left
                    > 0 if right
    Returns: shifted image
    """
    transform = AffineTransform(translation=(dx, dy))
    return warp(img, transform)


def perspective_transform(img):
    # 1. Get 4 points (nearish the edges, randomly generated)
    # 2. Order: top-left, top-right, bottom-right, bottom-left
    return four_point_transform(img)
