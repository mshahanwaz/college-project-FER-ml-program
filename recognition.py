import threading
from keras.models import load_model
from keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import serial
import requests
import time
from playsound import playsound
import os

# Paths
MODEL_PATH = "FER_Model.h5"
AUDIO_FILE_PATH = "/hardware/sound/alert.wav"
USB_PORT = "/dev/ttyUSB0"
FER_SERVER_URL = "https://fer-emotions.onrender.com/fer"

# Classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifier = load_model(MODEL_PATH)

# Number of emotions as labels
class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
currEmotion = False

# arduino = serial.Serial(USB_PORT, 115200)
cap = cv2.VideoCapture(0)

# API POST request to server
def post_function(emotionObject):
    requests.post(FER_SERVER_URL, json=emotionObject)

# Alert the user
def alert():
    audio_file = os.getcwd() + AUDIO_FILE_PATH
    playsound(audio_file)

while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float")/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            pos = preds.argmax()
            label = class_labels[pos]
            label_position = (x, y)

            time_stamp = time.time() * 1000
            emotionObject = { "emotion": label, "timestamp": time_stamp }
            x = threading.Thread(target=post_function, args=(emotionObject,))
            y = threading.Thread(target=alert)

            if not currEmotion:
                currEmotion = label
                x.start()
            else:
                if currEmotion != label:
                    currEmotion = label
                    x.start()
                    y.start()

            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # arduino.write((str(pos)).encode())
        else:
            cv2.putText(frame, "No Face Found", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # arduino.write((str(5)).encode())
    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
