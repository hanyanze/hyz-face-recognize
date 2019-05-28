# -*- coding: utf-8 -*-
# 人脸比较：将两张人脸图片进行对比
# 将两者之前的相似值进行打印
# 阈值0.6，阈值越小，条件越苛刻
import cv2
import face_recognition

# 加载被比较的图像
source_image = face_recognition.load_image_file("known/Benedict.jpg")
# 加载测试图像
compare_image = face_recognition.load_image_file("known/Martin.jpg")
# compare_image = face_recognition.load_image_file("unknown/unknown1.jpg")

# 获取人脸位置并做单人脸容错处理
source_locations = face_recognition.face_locations(source_image)
if len(source_locations) != 1:
    print("注意：图像一只能有一张人脸哦！")
    exit(0)
# 获取人脸位置并做单人脸容错处理
compare_locations = face_recognition.face_locations(compare_image)
if len(compare_locations) != 1:
    print("注意：图像二只能有一张人脸哦！")
    exit(0)

# 绘制图像一的人脸
for (top, right, bottom, left) in source_locations:
    print(top, right, bottom, left)
    cv2.rectangle(source_image, (left, top), (right, bottom), (0, 255, 0), 2)
# 绘制图像二的人脸
for (top, right, bottom, left) in compare_locations:
    print(top, right, bottom, left)
    cv2.rectangle(compare_image, (left, top), (right, bottom), (0, 255, 0), 2)


# 获取图像一的面部编码
source_face_encoding = face_recognition.face_encodings(source_image)[0]
# print("---")
# print(len(source_face_encoding))
source_encodings = [
    source_face_encoding,
]
# 获取图像二的面部编码
compare_face_encoding = face_recognition.face_encodings(compare_image)[0]

# 显示两幅得到人脸后的图像
source_image = source_image[:, :, ::-1]
cv2.imshow("image", source_image)
cv2.waitKey(0)
compare_image = compare_image[:, :, ::-1]
cv2.imshow("image", compare_image)
cv2.waitKey(0)

# 查看图像一与面部二的比较结果，阈值0.6，越小越苛刻
face_distances = face_recognition.compare_faces(source_encodings, compare_face_encoding, 0.6)
# print(face_distances)
# 输出结果
print("正常阈值为0.6时，测试图像是否与已知图像{}匹配的!".format("是" if face_distances else "不是"))

