import numpy as np
import cv2

class LineFollowingUtils:
    """
    Utils for lab2 (finding center of gravity)
    """
    def __init__(self):
        pass

    def center_of_gravity(self, image, threshold=100, bottom_pixels=100):
        bin = self.threshold_image(image, threshold)
        img_h = image.shape[0]
        x, y = self._center_of_grav(bin[img_h - bottom_pixels:img_h, :], threshold)
        return x, y + img_h - bottom_pixels

    def threshold_image(self, image, threshold):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bin = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY)
        return bin
    
    def _center_of_grav(self, image, threshold):
        x_sum = 0
        y_sum = 0
        area = 0

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if image[y][x] == 0:
                    area += 1
                    x_sum += x
                    y_sum += y

        if area == 0:
            return 0, 0

        # cv2.circle(image, (x_sum / area, y_sum / area), 20, (255, 0, 0), thickness=20)
        # cv2.imshow('Frame', image)
        # if cv2.waitKey(25):
        #     pass
        return x_sum / area, y_sum / area
