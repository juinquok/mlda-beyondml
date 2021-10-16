from VideoCapture import VideoCapture
from numpy import array
import os
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
import numpy as np
import pyttsx3

WORK_DIR = os.path.expandvars(os.environ.get("PWD"))
CLASS_LABELS = array(
    ['giddy', 'few', 'months', 'uncomfortable', 'no', 'room', 'spin', 'allergic'])

mediapipe = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
interpreter = Interpreter(model_path=WORK_DIR + "/medicine.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# TT2Speech
tts = pyttsx3.init()  # for tts
tts.setProperty('rate', 85)

predictions = []
sequence = []
threshold = 0.7


def mediapipe_detection(mediapipe, input_img):
    image = cvtColor(input_img, COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mediapipe.process(image)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


def tflite_predict(interpreter, sequence, input_details, output_details):
    input_arr = np.array(np.expand_dims(sequence, axis=0), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


print("Starting Threads")
video = VideoCapture().startCapture()
print("Video Thread Started!")

tts.say("Inference Start")
tts.runAndWait()

while True:
    CURRENT_IMAGE = np.empty(0)
    CURRENT_IMAGE = video.readImage()
    if not np.any(None):
        input_img = CURRENT_IMAGE
        keypoints = mediapipe_detection(mediapipe, input_img)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            res = tflite_predict(interpreter, sequence,
                                 input_details, output_details)[0]
            predicted_class_label = np.argmax(res)
            print("Pred: " + str(predicted_class_label))
            # Only give response if the model is confident
            if (len(predictions) == 0 or predictions[-1] != predicted_class_label) and res[predicted_class_label] > threshold:
                predictions.append(predicted_class_label)
                print(CLASS_LABELS[predicted_class_label])
                tts.say(CLASS_LABELS[predicted_class_label])
                tts.runAndWait()


# TODO: Implement AWS IPC Utils
