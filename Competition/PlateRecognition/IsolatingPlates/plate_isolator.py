#!/usr/bin/env python3

import cv2
import numpy as np


class PlateIsolator:
    """
    The goal of this module is to pick out parking and license plates

    Input: a clean image of the plates
           Random, likely terrible images picked up from Anki camera

    Resources:
    https://pysource.com/2018/06/05/object-tracking-using-homography-opencv-3-4-with-python-3-tutorial-34/
    https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html
    https://docs.google.com/document/d/1trqdpvf9x_Ft62-yL35qbelQnErPKc49posviM9nw4U/edit

    https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/
    """

    def __init__(self, feature_img):
        """
        Sets up our sift recognition based off of our pattern
        """
        self.MAX_IMG_WIDTH = 500

        self.feature_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2GRAY)
        self.rescale_feature_img()

        self.sift = cv2.xfeatures2d.SIFT_create()  # makes SIFT object

        # finds keypoints and gets descriptors in one method!
        self.key_points, self.descriptor = \
            self.sift.detectAndCompute(self.feature_img, None)
        
        # feature matching
        self.index_params = dict(algorithm=0, trees=5)
        self.search_params = dict()
        self.flann = cv2.FlannBasedMatcher(self.index_params,
                                           self.search_params)


    def show_ref(self, duration_ms=5000):
        """
        Use for debugging purposes: show our reference image

        TODO: add functionality that displays key points for visual ref
        """
        cv2.imshow("Features image", self.feature_img)
        cv2.waitKey(duration_ms)
    
    def show_ref_and_keypoints(self, duration_ms=5000):
        """
        For debugging purposes: figure out what these keypoints look like!
        """
        img = cv2.drawKeypoints(self.feature_img, self.key_points,
                                self.feature_img,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow("Features image with keypoints", img)
        cv2.waitKey(duration_ms)

    def rescale_feature_img(self):
        width = self.feature_img.shape[1]
        height = self.feature_img.shape[0]
        if (width > self.MAX_IMG_WIDTH):
            scale_factor = self.MAX_IMG_WIDTH / width
            dim = (self.MAX_IMG_WIDTH, int(height * scale_factor))
            self.feature_img = cv2.resize(self.feature_img, dim)

    def detectFeature(self, frame, greyframe):
        """
        Method: courtesy of homeography lab
        """
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(greyframe,
                                                                  None)
        matches = self.flann.knnMatch(self.descriptor, desc_grayframe, k=2)

        good_points = []

        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        query_pts = np.float32([self.key_points[m.queryIdx].pt for m in
                                good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in
                                good_points]).reshape(-1, 1, 2)

        if len(query_pts) == 0 or len(train_pts) == 0:
            return None

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC,
                                          5.0)

        if matrix is None:
            return None

        # matches_mask = mask.ravel().tolist()

        # perspective transform
        h, w = self.feature_img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        # print("pts: {}".format(pts))
        # print("matrix: {}".format(matrix))
        dst = cv2.perspectiveTransform(pts, matrix)

        # display result to screen
        homography = cv2.polylines(frame, [np.int32(dst)], True,
                                   (255, 0, 0), 3)
        cv2.imshow("Homography", homography)
        cv2.waitKey(5000)

        return np.int32(dst)

