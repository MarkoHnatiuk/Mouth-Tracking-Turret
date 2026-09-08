import cv2
import numpy as np
from helper import *
import pandas as pd
import serial
import time
from matplotlib import pyplot as plt





face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

arduino = serial.Serial(port='COM9', baudrate=9600, timeout=.1)


# Start webcam
cap = cv2.VideoCapture(0)

face_img = None
prediction = np.zeros([128,128])
origin_shape = [250,250]
face_x,face_y = 0,0
mouth_x, mouth_y = 0,0
width = 640
height = 480
margin=0.4
mouth_found = False
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Detect faces
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=20)
    # Show the main webcam feed
    cv2.imshow('Webcam Feed', frame)
    if len(faces) > 0:
        faceDetected = True
        (x, y, w, h) = faces[0]

        margin_x = int(w * margin)  # margin for width
        margin_y = int(h * margin)  # margin for height

        # Adjust the coordinates to include the margin
        x = max(x - margin_x, 0)
        y = max(y - margin_y, 0)
        w = min(w + 2 * margin_x, frame.shape[1] - x)
        h = min(h + 2 * margin_y, frame.shape[0] - y)

        face_x, face_y= x,y
        # Crop the face from the original frame
        face_img = frame[y:y + h, x:x + w]
        origin_shape = face_img.shape
        face_img = cv2.resize(face_img, (128, 128))
        # Optionally display the cropped face in another window
    else:
        faceDetected = False
        face_img = np.zeros([128,128,1])

    if faceDetected:
        input = np.expand_dims(face_img, axis=0)  # add batch dimension
        # predict the mouth position using the model
        prediction = modelLoading("unet_model_2.keras").predict(input)
        prediction = np.squeeze(prediction)
        prediction = (prediction > 0.5).astype(np.uint8)
        # show face with mouth mask
        prediction = prediction * 255
        face_with_prediction = np.maximum(face_img, prediction)
        cv2.imshow('Cropped Face', face_with_prediction)
        # calculate the middle of the mouth
        prediction = cv2.resize(prediction, origin_shape)
        mouth_x,mouth_y = face_x+getCenterMask(prediction)[0], face_y+getCenterMask(prediction)[1]
        if getCenterMask(prediction)[0]>0 and getCenterMask(prediction)[1]>0:
            mouth_found = True
        else:
            mouth_found = False
        # calculate the distance between the middle lines and middle of the mouth
        distance_x = width/2 - mouth_x
        distance_y = height/2 - mouth_y
        # show middle of the mouth
        plt.figure(figsize=(6, 6))
        plt.imshow(frame)
        plt.plot(mouth_x, mouth_y, 'ro')  # red dot at the centroid
        plt.text(mouth_x + 5, mouth_y, f'({mouth_x},{mouth_y})', color='red')
        plt.axis('off')
        plt.show()
        if mouth_found:
            # send the distance to arduino
            send_command(arduino,distance_x,distance_y)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        send_command(arduino,360,360)
        break


