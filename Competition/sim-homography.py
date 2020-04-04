import data_collection_utils as dcu
import image_processing_utils as ipu
import glob
import cv2
import numpy as np
import copy
import global_variables as gv

images = glob.glob(gv.path + '/Image_Data/*.png')

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    #  top_down = ipu.transform_to_top_down(edges)

    topdown = ipu.transform_to_top_down(img)

    gray_td = cv2.cvtColor(topdown, cv2.COLOR_BGR2GRAY)
    ret, thresh_td = cv2.threshold(gray_td, 210, 255, cv2.THRESH_BINARY)
    edges_post_td = cv2.Canny(thresh_td, 50, 150, apertureSize=3)

    cv2.imshow("img", img)
    cv2.imshow("topdown", topdown)
    cv2.imshow("edges_post_td", edges_post_td)

    threshold = 125

    lines = ipu.get_hough_lines(img=edges_post_td, img_type="edges", threshold=threshold)

    if lines is not None:
        ipu.draw_lines(topdown, lines, color=(0, 0, 255))
        edges_td_lines = edges_post_td * 128
        edges_td_lines = cv2.cvtColor(edges_td_lines, cv2.COLOR_GRAY2BGR)

        ipu.draw_lines(edges_td_lines, lines, color=(0, 0, 255))
        cv2.imshow("lines_edges", edges_td_lines)

    else:
        print("No Lines")

    cv2.imshow("lines", topdown)

    cv2.waitKey(0)

cv2.destroyAllWindows()
