#!/usr/bin/env python

import string
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw

path = os.path.dirname(os.path.realpath(__file__)) + "/"

"""
Plate generation is adapted from Miti's code (displayed below commented out)

Goal is to generate plates with every letter / number in each possible position
"""
pic_num = 0

for i in range(0, 200):

    # will give us every letter in both 1st and 2nd plate position
    # will give us every number in [0, 9]  in 1st and 2nd plate position
    plate_alpha = string.ascii_uppercase[i % 26] + string.ascii_uppercase[i % 26]
    num = (i % 10)*10 + (i % 10)
    plate_num = "{:02d}".format(num)

    # Write plate to image
    blank_plate = cv2.imread(path+'blank_plate.png')

    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)

    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = \
        ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
                           200)
    draw.text((48, 50), plate_alpha + " " + plate_num, (255, 0, 0), 
              font=monospace)

    # Convert back to OpenCV image and save
    blank_plate = np.array(blank_plate_pil)

    # Write license plate to file
    cv2.imwrite(os.path.join(path + "letter_pictures/",
                             "plate_{}{}_{}.png".format(plate_alpha, plate_num, pic_num)),
                blank_plate)
    
    pic_num += 1
