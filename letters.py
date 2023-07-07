import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("ModelLetter/keras_model.h5", "ModelLetter/labels.txt")

offset = 20
imgSize = 128

labels = ["B", "C", "L", "O", "T", "V", "W", "A"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    cv2.waitKey(1)

    if hands:
        hand1 = hands[0]
        x1, y1, w1, h1 = hand1['bbox']
        h = h1
        w = w1
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

            imgCrop = img[yb - offset: yt + h + offset, xl - offset: xr + w + offset]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        impgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            grayscale = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            imgColor = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)  # Convert grayscale to color
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            grayscale = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            imgColor = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)  # Convert grayscale to color

        cv2.imshow("Gray", imgColor)
        prediction, index = classifier.getPrediction(imgColor, draw=False)  # Use the color image for prediction
        cv2.putText(img, labels[index], (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

        print(prediction, labels[index])
