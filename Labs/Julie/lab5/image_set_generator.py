import cv2
import os


class ImageSetGenerator:

    def __init__(self, input_path='enph353_cnn_lab/letter_pictures/',
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
        labels = self.get_labels(files)

        i = 0
        for plate in processed_plates:
            a, b, c, d = self.segment_plates(plate)
            letters[i] = (a, labels[i] + '_0')
            letters[i + 1] = (b, labels[i + 1] + '_1')
            letters[i + 2] = (c, labels[i + 2] + '_0')
            letters[i + 3] = (d, labels[i + 3] + '_1')
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
        return plate[:, 0:w], plate[:, w:2*w], plate[:, 2*w:3*w], \
            plate[:, 3*w:4*w]

    def get_labels(self, files):
        """ relies on file name format: plate_AA11.png"""
        labels = [None] * 4 * len(files)
        i = 0
        for name in files:
            labels[i] = name[6]
            labels[i + 1] = name[7]
            labels[i + 2] = name[8]
            labels[i + 3] = name[9]
            i = i + 4

        return labels

    def save_plate_segment(self, plate, character):
        cv2.imwrite("{} letter_{}.png".format(self.output_dir,
                    character), plate)

    def generate_letter_set(self):
        letters = self.get_letters(self.files_in_folder())
        for letter in letters:
            self.save_plate_segment(letter[0], letter[1])


def main():
    img_gen = ImageSetGenerator()
    img_gen.generate_letter_set()


if __name__ == "__main__":
    main()
