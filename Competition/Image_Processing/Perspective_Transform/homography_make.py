"""
@file copy_make_border.py
@brief Sample code that shows the functionality of copyMakeBorder
"""
import sys
from random import randint
import cv2 as cv


def main(argv):
    borderType = cv.BORDER_CONSTANT
    window_name = "copyMakeBorder Demo"

    imageName = argv[0] if len(argv) > 0 else 'lena.jpg'
    # Loads an image
    src = cv.imread("ref_img.png")
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: copy_make_border.py [image_name -- default lena.jpg] \n')
        return -1

    print('\n'
          '\t   copyMakeBorder Demo: \n'
          '     -------------------- \n'
          ' ** Press \'c\' to set the border to a random constant value \n'
          ' ** Press \'r\' to set the border to be replicated \n'
          ' ** Press \'ESC\' to exit the program ')

    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    top = int(0.5 * src.shape[0])  # shape[0] = rows
    bottom = 30
    left = int(0.35 * src.shape[1])  # shape[1] = cols
    right = left
    value = [255, 255, 255]

    dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)

    scale_percent = 30  # percent of original size
    width = int(dst.shape[1] * scale_percent / 100)
    height = int(dst.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(dst, dim, interpolation=cv.INTER_AREA)[: int(height * .88), :]

    print('Resized Dimensions : ', resized.shape)

    cv.imwrite("new_ref.jpg", resized)

    cv.imshow("Resized image", resized)
    cv.waitKey(0)


if __name__ == "__main__":
    main(sys.argv[1:])
