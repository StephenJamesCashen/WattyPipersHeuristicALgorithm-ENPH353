import anki_vector as av
import cv2
import numpy as np
from anki_vector.util import degrees, distance_mm, speed_mmps
# from PIL import Image


class BlockTracker:

    def __init__(self, img_path="block_pattern.jpg"):
        ANKI_SERIAL = '0090452f'
        ANKI_BEHAVIOR = \
            av.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY

        # initialise our sift stuff
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("our image", self.img)
        # cv2.waitKey(1000)

        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp_image, self.desc_image = self.sift.detectAndCompute(self.img,
                                                                    None)

        # feature matching
        self.index_params = dict(algorithm=0, trees=5)
        self.search_params = dict()
        self.flann = cv2.FlannBasedMatcher(self.index_params,
                                           self.search_params)

        # initialise and run Watty Ember
        with av.Robot(serial=ANKI_SERIAL,
                      behavior_control_level=ANKI_BEHAVIOR) as robot:
            self.robot = robot
            self.img_width = 0
            self.last_error = 0
            self.closeness_threshold = 12000
            self.closeness_range = 1000
            self.def_speed_mmps = 20
            self.robot.camera.init_camera_feed()
            self.setupBot()
            # self.recite_hamlet()
            self.track_box()

    def fetch_image(self):
        frame = self.robot.camera.latest_image.raw_image

        np_frame = np.array(frame)

        while (not frame):
            cv2.waitKey(10)

        if not self.img_width:
            self.img_width = np_frame.shape[1]

        return np_frame, cv2.cvtColor(np_frame,
                                      cv2.COLOR_BGR2GRAY)

    def locate_box(self):
        pass

    def track(self, dst):
        if dst is None:
            if self.last_error > 0:
                self.robot.behavior.turn_in_place(degrees(-30))
            else:
                self.robot.behavior.turn_in_place(degrees(30))
            cv2.waitKey(1500)
        else:
            x, y, closeness = self.get_block_centre(dst)
            error = x - self.img_width / 2
            self.last_error = error

            if abs(error) < 50:
                self.approach_box(closeness)

            self.robot.behavior.turn_in_place(degrees(-error / 10))
            cv2.waitKey(500)

        # print("destination: \n{}".format(dst))
        # self.robot.behavior.say_text("Are you my mummy?")

    def approach_box(self, closeness):
        if abs(closeness - self.closeness_threshold) > self.closeness_range:
            # error positive if too far, negative if too close
            error = (self.closeness_threshold -
                     closeness) * 4 / self.closeness_range
            print(error)
            self.robot.behavior.drive_straight(distance_mm(error),
                                               speed_mmps(self.def_speed_mmps))

    def track_box(self):
        while True:
            frame, greyframe = self.fetch_image()
            cv2.imshow("robot view", greyframe)
            cv2.waitKey(1)
            dst = self.detectFeature(frame, greyframe)
            self.track(dst)

    def detectFeature(self, frame, greyframe):
        kp_grayframe, desc_grayframe = self.sift.detectAndCompute(greyframe,
                                                                  None)
        matches = self.flann.knnMatch(self.desc_image, desc_grayframe, k=2)

        good_points = []

        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        query_pts = np.float32([self.kp_image[m.queryIdx].pt for m in
                                good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in
                                good_points]).reshape(-1, 1, 2)

        if len(query_pts) == 0 or len(train_pts) == 0:
            return None

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC,
                                          5.0)

        if matrix is None:
            return None

        matches_mask = mask.ravel().tolist()

        # perspective transform
        h, w = self.img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        # print("pts: {}".format(pts))
        # print("matrix: {}".format(matrix))
        dst = cv2.perspectiveTransform(pts, matrix)

        # display result to screen
        homography = cv2.polylines(frame, [np.int32(dst)], True,
                                   (255, 0, 0), 3)
        cv2.imshow("Homography", homography)

        return np.int32(dst)

    def get_block_centre(self, dst):
        dst = dst[:, 0, :]
        dst_x = dst[:, 0]
        dst_y = dst[:, 1]

        x = np.sum(dst_x) / dst_x.shape[0]
        y = np.sum(dst_y) / dst_y.shape[0]

        x_arr = np.array([x for a in dst_x])
        y_arr = np.array([y for a in dst_y])

        x_diff = np.subtract(x_arr, dst_x)
        y_diff = np.subtract(y_arr, dst_y)

        closeness = np.sum(np.square(x_diff)) + np.sum(np.square(y_diff))

        return x, y, closeness

    def setupBot(self):
        self.robot.behavior.set_head_angle(degrees(0))
        self.robot.behavior.set_lift_height(1)

    def recite_hamlet(self):
        str = "To be or not to be, that is the question. \
                Whether tis nobler in the mind to \
                suffer the slings and arrows of outrageous fortune, \
                or to take arms against a sea of troubles \
                and by opposing end them. \
                To die, to sleep no more, and by a sleep to say we end the \
                thousand \
                natural shocks to which flesh is heir. Tis a consummation\
                devoutly \
                to be wished. To die, to sleep. To sleep perchance to dream. \
                Aye, there's the rub. For in this sleep of death what \
                dreams may \
                come once we have shuffled off this mortal coil must give us \
                pause. \
                There's the respect that makes calamity of so long life."
        self.robot.behavior.say_text(str)


def main():
    tracker = BlockTracker()


if __name__ == "__main__":
    main()
