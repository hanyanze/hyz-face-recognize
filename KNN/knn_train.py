# 训练k近邻分类器
import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    训练k近邻分类器进行人脸识别。
    param train_dir：包含每个已知人员的子目录及其名称的目录。
    目录结构：
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    param model_save_path :(可选）将模型保存在磁盘上的路径
    param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    param n_neighbors :(可选）在分类中称重的邻居数。如果未指定，则自动选择
    param knn_algo :(可选）支持knn.default的底层数据结构是ball_tree
    param verbose：训练的冗长
    return：返回在给定数据上训练的knn分类器。
    """
    X = []
    y = []

    # 循环遍历训练集中的每个人
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):# 如果train_dir/class_dir不是一个目录的话，就continue
            continue

        # 循环浏览当前人的每个训练图像
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # 如果训练图像中没有人（或人太多），请跳过图像。
                if verbose:
                    print("{} 这张图像不适合训练: {}。".format(img_path, "因为它上面没找到人脸" if len(face_bounding_boxes) < 1 else "因为它不止一张人脸"))
            else:
                # 将当前图像的面部编码添加到训练集
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    # 确定KNN分类器中用于加权的邻居数
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X)))) # 面部编码长度开平方后四舍五入取整数
        if verbose:
            print("自动选择n_neighbors:", n_neighbors)

    # 创建并训练KNN分类器
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # 保存训练后的KNN分类器
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


if __name__ == "__main__":
    # 训练的KNN分类，并将其保存到磁盘
    print("训练KNN分类器...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("训练完成！")