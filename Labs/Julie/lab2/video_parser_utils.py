import cv2

class VideoParserUtils:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
    
    def get_next_frame(self):
        return self.cap.read()
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()