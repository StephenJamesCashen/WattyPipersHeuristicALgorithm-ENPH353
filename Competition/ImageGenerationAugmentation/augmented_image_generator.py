#!/usr/bin/env python
import cv2
import augmentation_utils_beta as util
import os

ORIGINAL_IMAGE_DIR = 'Inputs/'
SAVE_IMAGE_DIR = 'Outputs/'
COMMON_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"

MODS_PER_IMAGE = 20

DEBUG = True

input_path = COMMON_PATH + ORIGINAL_IMAGE_DIR
output_path = COMMON_PATH + SAVE_IMAGE_DIR


def input_names():
    return os.listdir(input_path)


def input_images():
    return [cv2.imread("{}{}".format(input_path,
            file)) for file in input_names()]


def main():
    image_count = 0

    for img in input_images():
        # save any input image into our output set
        cv2.imwrite("{}image_{}.jpg".format(output_path, image_count), img)
        image_count += 1

        if DEBUG:
            cv2.imshow("img", img)
            cv2.waitKey(500)

        # generate and save augmentations
        for i in range(MODS_PER_IMAGE):
            new_img = util.randomise_augmentation(img)
            if DEBUG:
                cv2.imshow("img", new_img)
                cv2.waitKey(500)

            cv2.imwrite("{}image_{}.jpg".format(output_path, image_count),
                        new_img)
            image_count += 1


if __name__ == "__main__":
    main()
