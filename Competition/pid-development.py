import Competition.data_collection_utils as dcu
import Competition.image_processing_utils as ipu
import glob
import cv2
import numpy as np
import copy
import Competition.global_variables as gv

images = glob.glob(gv.path + '\\Image_Data\\image_*.png')

for fname in images:

    img = cv2.imread(fname)
    c_img = dcu.undistort(img)

    height = c_img.shape[0]
    width = c_img.shape[1]

    new_img = copy.copy(c_img)

    new_img[0:int(2*height/3),::] = 0


    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(new_img, 50, 150, apertureSize=3)

    top_down = ipu.transform_to_top_down(edges)
    cv2.imshow("edges", edges)
    cv2.imshow("cimg", top_down)

    # c_img = c_img[int(height/3):height,::]

    threshold = 100

    lines = ipu.get_hough_lines(img=top_down, threshold=threshold)
    if lines is not None:
        lines = list(filter(lambda line: not (5*np.pi/8 > line[0][1]> 3*np.pi/8), lines))

    ipu.draw_lines(top_down, lines)

    cv2.imshow("lines", c_img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()