from threading import Thread
from cv2 import VideoCapture
import MainRunner as runner


class VideoCapture:

    def __init__(self, src=0):
        self.stream = VideoCapture(src)
        self.stopFlag = False
        self.frame_seq = []
        (self.grabbed, self.frame) = self.stream.read()

    def startCapture(self):
        Thread(target=self.getFrames, args=()).start()
        return self

    def getFrames(self):
        while not self.stopFlag:
            if not self.grabbed:
                self.stopCapture()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                runner.CURRENT_IMAGE = self.frame

    def stopCapture(self):
        self.stopFlag = True
