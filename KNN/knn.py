"""
这是使用k近邻（KNN）算法进行人脸识别的示例。
When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.
Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.
For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.
* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.
用法：
1.准备一组您想要识别的已知人物的图像。在单个目录中组织图像每个已知人员的子目录。
2.然后，使用适当的参数调用“train”函数。如果你确定要传递'model_save_path'想要将模型保存到磁盘，
  以便您可以重新使用该模型而无需重新训练它。
3.调用“predict”并传入训练有素的模型，以识别未知图像中的人物。

注意：此示例需要安装scikit-learn！你可以用pip安装它：
$ pip3 install scikit-learn
"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


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


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    使用训练后的KNN分类器识别给定图像中的面部
    param X_img_path：要识别的图像的路径
    param knn_clf :(可选）一个knn分类器对象。如果未指定，则必须指定model_save_path。
    param model_path :(可选）pickle knn分类器的路径。如果未指定，则model_save_path必须为knn_clf。
    param distance_threshold :(可选）面部分类的距离阈值。它越大，机会就越大，将一个不知名的人误分类为已知人。
    图像中已识别面部的名称和面部位置列表：[（名称，边界框），...]。对于未被识别人员的面孔，将返回“未知”的名称。
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("无效的图片路径: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("必须提供knn分类器thourgh knn_clf或model_path")

    # 加载训练后的KNN模型（如果传入了一个）
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 加载图像文件并查找面部位置
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    # X_face_locations = face_recognition.face_locations(X_img, number_of_times_to_upsample=0, model="cnn")

    # 如果图像中未找到面，则返回空结果。
    if len(X_face_locations) == 0:
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


def show_prediction_labels_on_image(img_path, predictions):
    """
    直观地显示面部识别结果。
    param img_path：要识别的图像的路径
    param预测：预测函数的结果
    返回
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # 使用Pillow模块在脸部周围画一个方框
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Pillow中有一个错误，它会以非UTF-8文本爆炸
        # 使用默认位图字体时
        name = name.encode("UTF-8")

        # 在脸部下面画一个名称标签
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # 根据Pillow文档从内存中删除绘图库
    del draw

    # 显示生成的图像
    pil_image.show()


if __name__ == "__main__":
    # 第1步：训练的KNN分类，并将其保存到磁盘
    # 训练并保存模型后，您可以在下次跳过此步骤。
    print("训练KNN分类器...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("训练完成！")

    # 步骤2：使用训练有素的分类器，对未知图像进行预测
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("在{}中寻找面孔".format(image_file))

        # 使用训练后的分类器模型查找图像中的所有人
        # 注意：您可以传入分类器文件名或分类器模型实例
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # 在控制台上打印结果
        for name, (top, right, bottom, left) in predictions:
            print("- 找到了 {} 在 ({}, {})".format(name, left, top))

        # 显示覆盖在图像上的结果
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)