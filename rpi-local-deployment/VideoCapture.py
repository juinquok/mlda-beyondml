from threading import Thread
import cv2


class VideoCapture:

    def __init__(self, src=0):
        self.stopFlag = False
        self.stream = None
        self.grabbed = True
        self.frame = None

    def startCapture(self):
        Thread(target=self.getFrames, args=()).start()
        return self

    def getFrames(self):
        global CURRENT_IMAGE
        self.stream = cv2.VideoCapture(0)
        while self.stream.isOpened():
            while not self.stopFlag:
                if not self.grabbed:
                    self.stopCapture()
                else:
                    (self.grabbed, self.frame) = self.stream.read()
                    CURRENT_IMAGE = self.frame

    def stopCapture(self):
        self.stopFlag = True
        self.stream.release()

    def readImage(self):
        # print("Frame: " + self.frame)
        return self.frame
