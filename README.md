# hyz-face-recognize
========<br>
2019年5月28日，基于face-recognice实现人脸识别代码。<br>
根据 https://github.com/hanyanze/face_recognition 改写。<br>
谢谢大佬们无偿的开源。<br>
首先，在运行该项目前需要：<br>
```python
pip install face-recognice
```
该项目中共有六个（划掉）七个代码：<br>

* face-lacation-pic.py 是将一张照片中的面部都圈出来。<br>
* face-find-pic.py 是在人脸库中找到是否有需要检测到的人。<br>
* face-compare-pic.py 是利用两张照片进行比较是否是同一个人。<br>
* face-lacation.py 是利用USB免驱摄像头+OpenCV进行实时的面部检测。<br>
* 在KNN目录下：<br>
  * knn.py 直接将人脸库tarin训练成模型，然后直接检测test下的照片是谁。<br>
  可直接运行。<br>
  * knn_train.py 训练人脸库tarin。<br>
  * knn_predict.py 检测摄像头拍到的人脸（在knn_train之后运行）。<br>
 
好了，就酱紫，么么哒~~~~
