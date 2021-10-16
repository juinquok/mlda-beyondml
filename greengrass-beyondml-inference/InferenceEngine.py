from threading import Thread
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
import numpy as np
import pyttsx3


class InferenceEngine:

    def __init__(self, class_labels, camera, model_path="/home/pi"):
        self.mediapipe = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.interpreter = Interpreter(
            model_path=model_path + "/medicine.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.predictions = []
        self.sequence = []
        self.threshold = 0.7
        self.labels = class_labels
        self.input_img = None
        self.stopFlag = False
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 90)
        self.video = camera

    def startPrediction(self):
        prediction_thread = Thread(target=self.predict, args=()).start()
        return prediction_thread

    def stopPrediction(self):
        self.stopFlag = True

    def predict(self):
        while True:
            CURRENT_IMAGE = np.empty(0)
            CURRENT_IMAGE = video.readImage()
            if not np.any(None):
                self.input_img = CURRENT_IMAGE
                keypoints = self.mediapipe_detection()
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                if len(self.sequence) == 30:
                    res = self.tflite_predict()[0]
                    predicted_class_label = np.argmax(res)
                    # Only give response if the model is confident
                    if (len(self.predictions) == 0 or self.predictions[-1] != predicted_class_label) and res[predicted_class_label] > threshold:
                        self.predictions.append(predicted_class_label)
                        print(CLASS_LABELS[predicted_class_label])
                        self.tts.say(CLASS_LABELS[predicted_class_label])
                        self.tts.runAndWait()

    def mediapipe_detection(self):
        image = cvtColor(self.input_img, COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mediapipe.process(
            image)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

    def tflite_predict(self):
        input_arr = np.array(np.expand_dims(
            self.sequence, axis=0), dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_arr)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        return output_data
