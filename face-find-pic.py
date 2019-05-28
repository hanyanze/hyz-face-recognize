# -*- coding: utf-8 -*-
# 查找人脸：查找图片中的人脸并标记出来
import cv2
import os
import face_recognition

file_name = []
knowm_faces = []

# 加载文件中的人脸库图像
image_dir = "known/"
for parent, dirnames, filenames in os.walk(image_dir):
    for filename in filenames:
        # print(filename)
        # 加载图像
        frame = face_recognition.load_image_file(image_dir + filename)
        face_bounding_boxes = face_recognition.face_locations(frame)
        if len(face_bounding_boxes) != 1:
            # 如果训练图像中没有人（或人太多），请跳过图像。
            print("{} 这张图像不适合训练: {}。".format(image_dir + filename, "因为它上面没找到人脸" if len(face_bounding_boxes) < 1 else "因为它不止一张人脸"))
        else:
        # encoding
            frame_face_encoding = face_recognition.face_encodings(frame)[0]
            # 加到列表里
            knowm_faces.append(frame_face_encoding)
            file_name.append(filename)

# 加载未知图像
frame = face_recognition.load_image_file("unknown/unknown1.jpg")
# encoding
frame_face_encoding = face_recognition.face_encodings(frame)[0]
# 比较获得结果
results = face_recognition.compare_faces(knowm_faces, frame_face_encoding)

print(results)

# 输出结果
if not True in results:
    print("默认人脸库中{}此人".format("无"))
else:
    for i in range(len(results)):
        if results[i]:
            name = file_name[i].rstrip(".png")
            name = name.rstrip(".jpg")
            print("匹配到的人脸为{}".format(name))



# 下面的是一个一个加载的，墨迹。
# frame1 = face_recognition.load_image_file("image/1.jpg")
# frame2 = face_recognition.load_image_file("image/2.jpg")
# frame3 = face_recognition.load_image_file("image/3.jpg")
#
# frame1_face_encoding = face_recognition.face_encodings(frame1)[0]
# frame2_face_encoding = face_recognition.face_encodings(frame2)[0]
# frame3_face_encoding = face_recognition.face_encodings(frame3)[0]
#
# known_faces = [
#     frame1_face_encoding,
#     frame2_face_encoding,
#     frame3_face_encoding
# ]
#
# frame = face_recognition.load_image_file("unknown.jpg")
#
# frame_face_encoding = face_recognition.face_encodings(frame)[0]
#
# results = face_recognition.compare_faces(known_faces, frame_face_encoding)
#
# print(results)
# if not True in results:
#     print("默认人脸库中{}此人".format("无"))
# else:
#     for i in results:
#         有这个人，逻辑懒得写了



