#!/usr/bin/env python

class AugmentationUtils:
    """
    Image augmentation methods to get more plate images
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
    Rotates image and crops it so that there's no added data
    Params: phi - angle of rotation (radians)
    Returns: rotated image
    """
    def rotate(self, phi):
        pass

    """
    Add blur
    Params: dunno yet, some kind of number?
    Returns: blurred image
    """
    def blur(self):
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
