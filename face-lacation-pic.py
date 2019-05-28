# -*- coding: utf-8 -*-
# 查找人脸：查找人脸并标记出来
import cv2
import face_recognition

# 加载被比较的图像
frame = face_recognition.load_image_file("known/Benedict.jpg")

# 使用CPU获得人脸边界框的array
face_locations = face_recognition.face_locations(frame)
# 使用CNN利用GPU/CUDA加速获得人脸边界框的array
# 相对更准确
# face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn")

print("该张图片中有 {} 张人脸。".format(len(face_locations)))

# 圈出人脸边界框
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

# 显示得到人脸后的图像
frame = frame[:, :, ::-1]
cv2.imshow("image", frame)
cv2.waitKey(0)
