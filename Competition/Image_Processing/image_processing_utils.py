import cv2
import numpy as np
"""

"""


def get_hough_lines(img, threshold=125, show=False, waitKey=5, verbose=False,
                   probabalistic=False, minLineLength=100, maxLineGap=10) -> np.ndarray:
    """Generates outputs lines from an image using the Hough transform features built into open-cv.

    Arguments:
        img {open-cv formatted image} -- image to generate lines on

    Keyword Arguments:
        threshold {int} -- threshold to pass to the Houghlines generator. (default: {125})
        show {bool} -- if true will display the image with the lines. (default: {False})
        waitKey {int} -- time to display the image if show is true. (default: {5})
        verbose {bool} -- Whether to print information to the console (Currently unimplemented) (default: {False})
        probabalistic {bool} -- if true uses the probabalistic hough transform see cv2.HoughLinesP (default: {False})
        minLineLength {int} -- minimum length of a line, passed to cv2.HoughLinesP (default: {100})
        maxLineGap {int} -- maximum gap in a line, passed to cv2.HoughLinesP (default: {10})

    Returns:
        np.ndarray -- array of lines detected by the respective transforms.
                        If probabalistic is true, will return list of endpoints of each line segment.
                        Otherwise will return a point on each line and the angle of a vector perpindicular to the line (centered at the origin.)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    if probabalistic:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold, minLineLength, maxLineGap)

        if show:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow(img)
            cv2.waitKey()
            cv2.waitKey(waitKey)
    else:
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)

        if show:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow(img)
            cv2.waitKey(waitKey)
