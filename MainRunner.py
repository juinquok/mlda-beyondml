from InferenceEngine import InferenceEngine
from VideoCapture import VideoCapture

CURRENT_IMAGE = None

# Implement Main Method
video = VideoCapture().startCapture()
inference = InferenceEngine.startPrediction()

# TODO: Implement AWS IPC Utils
