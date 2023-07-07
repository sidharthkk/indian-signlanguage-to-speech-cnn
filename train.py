import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder="DataSet/DataLetters/A"
counter=0

labels=["Bird","Flower","Good","Sorry","Thank You"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand1 = hands[0]
        x1, y1, w1, h1 = hand1['bbox']
        h=h1
        w=w1
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
        impgCropShape =imgCrop.shape

            #imgWhite[0:imgCropShape[0],0:imgCropShape[1]] = imgCrop

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, : ] = imgResize

        cv2.imshow("ImageWhite", imgWhite)
        grayscale = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", grayscale)

    cv2.imshow("Image", img)
    key= cv2.waitKey(1)
    if key == ord("s") or key == ord("S"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',grayscale)
        print (counter)