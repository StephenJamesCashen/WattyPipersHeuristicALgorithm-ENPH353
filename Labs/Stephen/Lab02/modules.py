import cv2 as cv
import functools
class modules:

    def self.getCOM(img):
        y_len=len(img)
        x_len=len(img)

        total=0
        wsum=[0,0]
        for x in range(x_len):
            for y in range(y_len):
               wsum += img[x][y][x,y]
               total += img[x][y]

        com=wsum/total
        return com
