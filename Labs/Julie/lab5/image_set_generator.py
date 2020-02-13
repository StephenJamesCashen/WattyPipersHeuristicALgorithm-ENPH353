import cv2
import os


class ImageSetGenerator:

    def __init__(self, input_path='enph353_cnn_lab/pictures/',
                 output_path='outputs/', img_width=300):
        self.input_dir = os.path.dirname(os.path.realpath(__file__)) + \
            "/" + input_path
        self.output_dir = os.path.dirname(os.path.realpath(__file__)) + \
            "/" + output_path
        self.img_width = img_width

    def files_in_folder(self):
        # adapted from Miti's code
        """
        Returns list of strings where each entry is a file in our dir
        Assumes directory contains only pictures we want
        """
        img_list = os.listdir(self.input_dir)
        return img_list

    def get_letters(self, files):
        plates = [cv2.imread("{}{}".format(self.input_dir, file),
                  cv2.IMREAD_GRAYSCALE) for file in files]
        processed_plates = [self.process_image(plate) for plate in plates]
        letters = [None] * 4 * len(processed_plates)
        i = 0
        for plate in processed_plates:
            a, b, c, d = self.segment_plates(plate)
            letters[i] = a
            letters[i + 1] = b
            letters[i + 2] = c
            letters[i + 3] = d
            i = i + 4

        return letters

    def process_image(self, img):
        if img is None:
            print("image is none help help help")
            return
        ratio = self.img_width / img.shape[1]
        return cv2.resize(img, dsize=(self.img_width, int(img.shape[0] * ratio)),
                          interpolation=cv2.INTER_CUBIC)

    def segment_plates(self, plate):
        # hardcoded for now, find more better method
        w = int(plate.shape[1]/4)
        # cv2.imshow('hi', plate[:, 0:w])
        # cv2.waitKey(2000)
        # cv2.imshow('hi', plate[:, w:2*w])
        # cv2.waitKey(2000)
        # cv2.imshow('hi', plate[:, 2*w:3*w])
        # cv2.waitKey(2000)
        # cv2.imshow('hi', plate[:, 3*w:4*w])
        # cv2.waitKey(2000)
        return plate[:, 0:w], plate[:, w:2*w], plate[:, 2*w:3*w], \
            plate[:, 3*w:4*w]


def main():
    img_gen = ImageSetGenerator()
    letters = img_gen.get_letters(img_gen.files_in_folder())
    cv2.imshow('hi', letters[0])
    cv2.waitKey(10000)


if __name__ == "__main__":
    main()
