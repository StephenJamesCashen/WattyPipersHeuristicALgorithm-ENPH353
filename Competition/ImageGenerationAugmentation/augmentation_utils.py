#!/usr/bin/env python
import cv2
from skimage.transform import rotate


class AugmentationUtils:
    """
    Image augmentation methods to get more plate images
    Most code was sourced / modified from:
    https://github.com/govinda007/Images/blob/master/augmentation.ipynb

    Credit to original author
    """
    def __init__(self, img):
        self.img = img

    """
    Scales image
    Params: c_x - horizontal scaling factor (positive int)
           c_y - vertical scaling factor (positive int)
    Returns: scaled image
    """
    def scale(self, c_x, c_y):
        pass

    """
    Rotates image
    Params: phi - angle of rotation (degrees)
    Returns: rotated image
    """
    def rotate(self, phi):
        return rotate(self.img, angle=phi)

    """
    Adds uniform blur
    Params: dunno yet, size of Gaussian blurring kernel
    Returns: blurred image
    """
    def blur(self, kernel_size):
        pass

    """
    Adds motion blur
    Params: dunno yet
    Returns: motion blurred image
    """
    def motion_blur(self):
        pass

    """
    Add noise
    Params: dunno yet
    Returns: noisier image
    """
    def noise(self):
        pass

    """
    Change hue, saturation, brightness
    If any resulting pixels have invalid hsb, 
    the values are capped at 255 and 0
    Params: hue - fractional change of hue
            saturation - fractional change of saturation
    Returns: HSB modified image
    """
    def hsb(self, h, s, b):
        pass

    """
    Shifts image
    Params: dx - shift of the image in pixels in x
            dy - shift of image in pixels in y
                    < 0 if left
                    > 0 iif right
    Returns: shifted image
    """
    def shift(self, dx, dy):
        pass


