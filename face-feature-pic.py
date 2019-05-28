# 自动识别人脸特征
# 导入face_recogntion模块，可用命令安装 pip install face_recognition
import cv2
import numpy as np
import face_recognition

# 加载被比较的图像
frame = face_recognition.load_image_file("known/Benedict.jpg")

# 查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(frame, face_locations = None, model ='large')
# 查找图像中所有面部的鼻子、左眼、右眼面部特征
# face_landmarks_list = face_recognition.face_landmarks(frame, face_locations=None, model='small')

print("该张图片中有 {} 张人脸。".format(len(face_landmarks_list)))
# print(face_landmarks_list)

for face_landmarks in face_landmarks_list:

    #打印此图像中每个面部特征的位置
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

    # facial_features = [
    #     'nose_tip',
    #     'left_eye',
    #     'right_eye',
    # ]

    # 让我们在图像中描绘出每个人脸特征！
    for facial_feature in facial_features:
        pts = np.array(face_landmarks[facial_feature], np.int32) #数据类型必须是int32
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (0, 0, 255), 2)  # 图像，点集，是否闭合，颜色，线条粗细
# 显示得到人脸后的图像
frame = frame[:, :, ::-1]
cv2.imshow("image", frame)
cv2.waitKey(0)




