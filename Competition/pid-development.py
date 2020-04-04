import data_collection_utils as dcu
import image_processing_utils as ipu
import glob
import cv2
import numpy as np
import copy

import global_variables as gv
from functools import reduce

images = glob.glob(gv.path + '/Image_Data/*.png')

for fname in images:

    img = cv2.imread(fname)
    # c_img = dcu.undistort(img)
    c_img = img
    height = c_img.shape[0]
    width = c_img.shape[1]

    new_img = copy.copy(c_img)

    # new_img[0:int(2*height/3),::] = 0

    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    top_down = ipu.transform_to_top_down(img)
    cv2.imshow("td", top_down)
    # c_img = c_img[int(height/3):height,::]

    threshold = 100

    lines = ipu.get_hough_lines(img=edges, img_type="edges", threshold=threshold)

    cv2.imshow("original", c_img)

    cv2.imshow("edges", edges)

    int_img = copy.copy(c_img)
    # The horizon is the vanishing line for the ground plane.

    horizon = np.array([[np.round(height / 2), np.round(np.pi / 2, 6)]])
    intersections = []

    for line in lines:
        intersections.append(ipu.intersection(line, horizon))

    for intersection in intersections:
        int_img = cv2.drawMarker(int_img, (intersection[0], intersection[1]), (191, 116, 59),
                                 markerType=cv2.MARKER_STAR, thickness=2)
    ipu.draw_lines(int_img, lines)
    cv2.imshow("intersections", int_img)

    intersections_filtered = np.array(list(filter(lambda intersection: 0 <= intersection[0] <= 640, intersections)))
    if intersections_filtered.__len__() > 0:
        print(intersections_filtered[:, 0])
        error = reduce(lambda x, y: x + y, intersections_filtered[:, 0]) / intersections_filtered.__len__() - width / 2
        print(error)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
