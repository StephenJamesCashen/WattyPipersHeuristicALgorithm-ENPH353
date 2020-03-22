import cv2
import numpy as np
import Competition.global_variables as gv

def get_hough_lines(img, threshold=125, probabilistic=False, min_line_length=100, max_line_gap=10,
                    verbose=False, ) -> np.ndarray:
    """

    @param img: image to generate lines on
    @param threshold: threshold to pass to the Houghlines generator
    @param probabilistic: if true uses the probabalistic hough transform see cv2.HoughLinesP (
    @param min_line_length: minimum length of a line, passed to cv2.HoughLinesP
    @param max_line_gap: maximum gap in a line, passed to cv2.HoughLinesP
    @param verbose: Whether to print information to the console (Currently unimplemented)

    @return: ndarray of lines. By default each line is represented by the length and angle of a vector
    (with origin (0,0)) normal to and intersecting the line. If probabalistic is true lines are instead
    represented by a pair of end points of a line segment,

    @todo Correct line rendering for larger images
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    else:
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

    if probabilistic:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold, min_line_length, max_line_gap)

    else:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    return lines


def draw_lines(img, lines, probabilistic=False):
    for line in lines:
        if probabilistic:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        else:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def transform_to_top_down(img):
    """
    @param img:
    @return: transformed image
    """
    homography = np.load(gv.path + "\\Image_Processing\\Perspective_Transform\\homography.npy")
    shape = np.load(gv.path + "\\Image_Processing\\Perspective_Transform\\shape.npy")
    img = cv2.warpPerspective(img, homography, (shape[1], shape[0]))
    return img