#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 摄像头读取并识别人脸

import cv2
import face_recognition

cap=cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    face_locations = face_recognition.face_locations(frame)
    # face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    # 圈出人脸边界框
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # 显示
    cv2.imshow('image',frame)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()