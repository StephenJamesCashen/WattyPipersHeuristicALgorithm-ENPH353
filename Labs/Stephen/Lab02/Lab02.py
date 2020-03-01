import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time


def getCOM(img):
        total=0
        wsum=np.array([0,0])
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
               total = total + img[y][x]
               wsum = wsum + img[y][x]*np.array([y,x])
        if total == 0:
            return (0,0)
        else:com=wsum/total
        return com


Raw = cv.VideoCapture('raw_video_feed.mp4')

if __name__ == "__main__":
    while True:
        ret, frame = Raw.read()
        if ret == False:
            break
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        threshold = 100
        _, thresh = cv.threshold(grey, threshold, 255, cv.THRESH_BINARY_INV)
        
        bottom = thresh[140:]

        y_com, x_com = getCOM(bottom)
        cv.circle(frame,(x_com,y_com+140), 25, (0,0,160), -1)

        cv.imshow('frame', frame)
        keyboard = cv.waitKey(30)


    Output.release()

# ret, frame = Raw.read()
# grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# threshold = 100
# _, thresh = cv.threshold(grey, threshold, 255, cv.THRESH_BINARY)
# print(thresh[200])
