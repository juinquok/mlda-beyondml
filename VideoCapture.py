from threading import Thread
from cv2 import VideoCapture


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
                (curr_grab, curr_frame) = self.stream.read()
                if len(self.frame_seq) < 30:
                    self.frame_seq.append(curr_frame)
                else:
                    self.frame_seq = self.frame_seq[1:]
                    self.frame_seq.append(curr_frame)
                (self.grabbed, self.frame) = (curr_grab, curr_frame)

    def stopCapture(self):
        self.stopFlag = True
