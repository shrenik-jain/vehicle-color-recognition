#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

import matplotlib.pyplot as plt 
import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path

cap = cv2.VideoCapture('sample_videos/car_driving.mp4')
car_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_car.xml')
prediction = ''

while True:

    # Capture frame-by-frame
    (ret, frame) = cap.read()

    cv2.putText(frame,'Color: ' + prediction, (15, 45), cv2.FONT_HERSHEY_PLAIN, 3, 200, 3,)

    cars = car_classifier.detectMultiScale(frame, 1.4, 2)
    
    # Extract bounding boxes for any car identified
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cropped_image = frame[y+2:y+h-2, x+2:x+w-2]
        cv2.imwrite('sample_videos/car.png', cropped_image)


    # Display the resulting frame
    cv2.imshow('Car Color Detector', frame)

    color_histogram_feature_extraction.color_histogram_of_test_image('sample_videos/car.png')

    prediction = knn_classifier.main('training.data', 'test.data')

    #Quit when enter Key is hit
    if cv2.waitKey(1) == 13: 
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()		
