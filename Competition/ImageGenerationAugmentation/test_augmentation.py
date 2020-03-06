#!/usr/bin/env python
import cv2
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
    # consider: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders


def test_blur(util):
    for i in range(1, 15):
        img = util.blur(i * 2 + 1)
        cv2.imshow("blur", img)
        cv2.waitKey(1000)


def test_motion_blur(util):
    for i in range(1, 15):
        img = util.motion_blur(i * 2 + 1)
        cv2.imshow("motion_blur", img)
        cv2.waitKey(1000)


def test_noise(util):
    modes = ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
    # Notes: salt, pepper, s&p are not as useful I think
    for mode in modes:
        img = util.noise(mode)
        cv2.imshow("noise", img)
        cv2.waitKey(1000)

    for i in range(10):
        img = util.rand_noise()
        cv2.imshow("noise", img)
        cv2.waitKey(1000)


def test_hsb(util):
    for i in range(10):
        img = util.hsb()
        cv2.imshow("hsb", img)
        cv2.waitKey(1000)


def test_shift(util):
    for dx in range(-2, 2):
        for dy in range(-2, 2):
            img = util.shift(dx * 50, dy * 50)
            cv2.imshow("shift", img)
            cv2.waitKey(1000)


def test_perspective_transform(util):
    for i in range(10):
        img = util.perspective_transform()
        cv2.imshow("perspective", img)
        cv2.waitKey(1000)


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
    test_perspective_transform(aug_util)
    print("Test complete: successful extermination")


if __name__ == "__main__":
    main()
