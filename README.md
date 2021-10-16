# mlda-beyondcovid
This project deploys a sign language machine learning model that converts hand actions into text that is played via text to speech. This conversion is achieved by first passing the frame through a Mediapipe model and then into a Tensorflow LSTM model. 

This model can then be converted to a TFLite model that is deployed to the edge for inference on a RPI device. This deployment is controlled by AWS IOT Greengrass.

## model-training
To run the code in this folder, pip install the following dependencies:
```
!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib pyttsx3
```

### keypoint_recorder.py
Code to generate model training data by defining actions and recording respective key points from the hands using Mediapipe. Key points saved as numpy array for each frame

### model_training.py
Code to train LSTM model used to prediction user's action. Model code snippet shown below,
```
# Define model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Implement early stopping
early_stopping = EarlyStopping(monitor='categorical_accuracy', min_delta=0.001, patience=100, verbose=1, mode='auto', restore_best_weights=True)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
```
Although simple, our model has the following benefits:
1. **Less data** is required for training, as relevant dataset does not exist.
2. **Faster training**, model has 203,624 trainable parameters.
3. **Faster detections** in real time, lightweight model allows for faster prediction, especially for edge devices with low computing resources.


### model_tester.py
Code to test the model on a live camera feed (current device) using OpenCV. Outputs predicted text on video feed and has text-to-speech implemented.


### tflite_converter.py
Converts .h5 model to .tflite for deployment to the edge. TensorFlow Lite is a set of tools that enables on-device machine learning by helping developers run their models on mobile, embedded, and IoT devices.

## rpi-local-deployment
### cvs_capture.py
This code is used for testing the RPI that also shows the output live on the RPI via the user interface.
### MainRunner.py
This code starts up the entire processing with the video capture being done on one thread while the inference runs on another thread.
### VideoCapture.py
This class is used to run the video capture in a thread of its own.
### InferenceEngine.py
This class can be used to run the inference in another thread. It is not used in the deployment currently. 

## greengrass-beyondml-inference.py
This is the deployment package that is deployed to the edge device using AWS IOT Greengrass.

### config_utils.py
This file holds the values that are initialized by the RPI on startup of the inference code as well as all constants.
### inference.py
This is the main method this is run by the RPI. It constantly checks the AWS backend via IPC to see if there is any update and then will update the running inference thread.
### InferenceEngine.py
This class is used for the inference code. It runs its own thread and gets the video frame from the provided video thread.
### VideoCapture.py
This class is used for the video capture and it exposes the readImage method to allow for other objects to read its frame.
### IPCUtils.py
This is a helper file for all AWS functionality exposed by the AWS IOT SDK.

## greengrass-cicd-serverless
This Lambda functions implement the serverless pipeline for deployment of the model after it is trained. The functions are used in conjunction with AWS EventBridge, S3 buckets and AWS CodePipeline to trigger deployments.
### deployment_update.py
This is the code for the Lambda function that updates the overall AWS Greengrass deployment that contains 2 components, the inference component and the model store component. This code will trigger a redeployment of the model and inference to all IOT devices on the network. 
### inference_update.py
This is the code for the Lambda function that updates the inference component. It will update the component version of the code to match the latest file in the S3 bucket. This is part of the CodePipeline and it will trigger the deployment update next. 
### model_update.py
This is the code for the Lmabda function that updates the model component. 