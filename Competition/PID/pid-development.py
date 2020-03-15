import anki_vector
import Competition.Data_Collection.data_collection_utils as dcu
import Competition.Image_Processing.image_processing_utils as ipu
import Competition.PID.pid_utils as pidu
import glob
import cv2
import numpy as np
import copy

PATH = "C:\\Users\\works\\OneDrive\\Personal_Documents\\University\\Courses\\ENPH 353\\enph353_git" \
       "\\WattyPipersHeuristicALgorithm-ENPH353\\"

images = glob.glob(PATH + 'Competition\\Image_Data\\image_*.png')

for fname in images:

    img = cv2.imread(fname)
    c_img = dcu.undistort(img, PATH + 'Competition\\Data_Collection\\Camera_Calibration')

    height = c_img.shape[0]
    width = c_img.shape[1]

    new_img = copy.copy(c_img)

    new_img[0:int(2*height/3),::] = 0


    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(new_img, 50, 150, apertureSize=3)


    cv2.imshow("edges", edges)
    cv2.imshow("cimg", c_img)

    # c_img = c_img[int(height/3):height,::]

    threshold = 100

    lines = ipu.get_hough_lines(img=new_img, threshold=threshold)
    if lines is not None:
        lines = list(filter(lambda line: not (5*np.pi/8 > line[0][1]> 3*np.pi/8), lines))

    ipu.draw_lines(c_img, lines)

    cv2.imshow("lines", c_img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()