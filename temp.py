#previouswords
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("ModelWord/keras_model.h5", "ModelWord/labels.txt")
engine = pyttsx3.init()

offset = 10
imgSize = 128

labels = ["Bird", "Flower", "Good", "Love", "Salute", "Sorry", "Thank You"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    cv2.waitKey(1)

    if hands:
        hand1 = hands[0]
        x1, y1, w1, h1 = hand1['bbox']
        imgCrop = img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset]

        if len(hands) == 2:
            hand2 = hands[1]
            x2, y2, w2, h2 = hand2['bbox']

            if y1 > y2:
                yb = y2
                yt = y1
            else:
                yb = y1
                yt = y2

            if h1 > h2:
                h = h1
            else:
                h = h2

            firstHandType = hand2["type"]

            if firstHandType == "Left":
                xl = x1
                xr = x2
                w = w2
            else:
                xl = x2
                xr = x1
                w = w1

            imgCrop = img[yb - offset:yt + h + offset, xl - offset:xr + w + offset]

        cv2.imshow("Crop", imgCrop)

        prediction, index = classifier.getPrediction(imgCrop, draw=False)
        cv2.putText(img, labels[index], (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.imshow("TempImage", img)

        spoken_word = labels[index]
        print(prediction, spoken_word)

        # Speak the predicted word
        engine.say(spoken_word)
        engine.runAndWait()

