#!/usr/bin/env python
import cv2
import math
from augmentation_utils import AugmentationUtils as au


def test_scale(util):
    for i in range(1, 5):
        for j in range(1, 5):
            img = util.scale(i / 6, j / 6)
            cv2.imshow("scale", img)
            cv2.waitKey(1000)


def test_rotate(util):
    for i in range(-3, 3):
        img = util.rotate(15 * i)
        cv2.imshow("rotate", img)
        cv2.waitKey(1000)


def test_blur(util):
    pass


def test_motion_blur(util):
    pass


def test_noise(util):
    pass


def test_hsb(util):
    pass


def test_shift(util):
    pass


def main():
    print("exterminate: being test")
    img = cv2.imread('dalek_test.jpeg')
    aug_util = au(img)
    test_scale(aug_util)
    test_rotate(aug_util)
    test_blur(aug_util)
    test_motion_blur(aug_util)
    test_noise(aug_util)
    test_hsb(aug_util)
    test_shift(aug_util)
    print("Test complete: successful extermination")


if __name__ == "__main__":
    main()
