import cv2
import numpy as np
import global_variables as gv


def get_hough_lines(img, img_type="edges", threshold=125, probabilistic=False, min_line_length=100, max_line_gap=10,
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
    if img_type == "color":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    elif img_type == "gray":
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
    elif img_type == "edges":
        edges = img
    else:
        raise ValueError("Img type " + img_type + " not supported")

    if probabilistic:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold, min_line_length, max_line_gap)

    else:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    return lines


def draw_lines(img, lines, probabilistic=False, color=(0, 0, 255), size=2):
    length = max((img.shape[0], img.shape[1]))

    for line in lines:
        if probabilistic:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, size)

        else:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + length * (-b))
                y1 = int(y0 + length * (a))
                x2 = int(x0 - length * (-b))
                y2 = int(y0 - length * (a))

                cv2.line(img, (x1, y1), (x2, y2), color, size)


from collections import defaultdict


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def find_intersections(lines, k_groups=2):
    segmented = segment_by_angle_kmeans(lines, k=k_groups)

    return segmented_intersections(segmented)


def transform_to_top_down(img, sim=True, shape=None):
    """
    @param sim:
    @param shape:
    @param img:
    @return: transformed image
    """
    if sim:
        homography = np.load(gv.path + "/Image_Processing/Perspective_Transform/homography-sim_v0.npy")
        if shape is None:
            shape = np.load(gv.path + "/Image_Processing/Perspective_Transform/shape-sim.npy")
    else:
        homography = np.load(gv.path + "/Image_Processing/Perspective_Transform/homography.npy")
        if shape is None:
            shape = np.load(gv.path + "/Image_Processing/Perspective_Transform/shape.npy")
    print(homography)
    print(shape)

    img = cv2.warpPerspective(img, homography, (shape[1], shape[0]))
    return img
