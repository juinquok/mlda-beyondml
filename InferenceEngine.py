from threading import Thread
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter
from cv2 import cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
import numpy as np
import MainRunner as runner


class InferenceEngine:

    def __init__(self, class_labels, model_path='/home/pi',):
        self.mediapipe = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.interpreter = Interpreter(model_path=model_path + "/model.tflite")
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

    def startPrediction(self):
        Thread(target=self.predict, args=()).start()
        return self

    def stopPrediction(self):
        self.stopFlag = True

    def predict(self):
        if runner.CURRENT_IMAGE != None and not self.stopFlag:
            self.input_img = runner.CURRENT_IMAGE
            keypoints = self.mediapipe_detection()
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:]
            if len(self.sequence) == 30:
                res = self.tflite_predict()[0]
                predicted_class_label = np.argmax(res)
                # Only give response if the model is confident
                if self.predictions[-1] != predicted_class_label and res[predicted_class_label] > self.threshold:
                    self.predictions.append(predicted_class_label)
                    print(self.labels[predicted_class_label])
                    # Now invoke TT2Speech

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
