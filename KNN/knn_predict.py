# 摄像头测试k近邻分类器
import os
import cv2
import os.path
import pickle
import face_recognition

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    使用训练后的KNN分类器识别给定图像中的面部
    param X_img：要识别的图像
    param knn_clf :(可选）一个knn分类器对象。如果未指定，则必须指定model_save_path。
    param model_path :(可选）pickle knn分类器的路径。如果未指定，则model_save_path必须为knn_clf。
    param distance_threshold :(可选）面部分类的距离阈值。它越大，机会就越大，将一个不知名的人误分类为已知人。
    图像中已识别面部的名称和面部位置列表：[（名称，边界框），...]。对于未被识别人员的面孔，将返回“未知”的名称。
    """

    if knn_clf is None and model_path is None:
        raise Exception("必须提供knn分类器thourgh knn_clf或model_path")

    # 加载训练后的KNN模型（如果传入了一个）
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 加载图像并查找面部位置
    X_face_locations = face_recognition.face_locations(X_img)
    # X_face_locations = face_recognition.face_locations(X_img, number_of_times_to_upsample=0, model="cnn")

    # 如果图像中未找到面，则返回空结果。
    if len(X_face_locations) == 0:
        print("没有检测到人脸！")
        return []

    # 在测试iamge中查找面部的编码
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # 使用KNN模型找到测试的最佳匹配
    # 找到一个点的K近邻。返回每个点的邻居的索引和距离。
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    # print(closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(are_matches)

    # 预测类并删除不在阈值范围内的分类
    # predict:返回分类的标签
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        predictions = predict(small_frame, model_path="trained_knn_model.clf")
        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
