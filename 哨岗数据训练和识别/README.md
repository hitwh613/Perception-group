# 数据训练和识别

## yoloV3介绍

我们决定采用YOLOv3算法。YOLOv3是一个既快速，又准确的实时目标检测系统。从YOLOv3的论文中我们可以看到在COCO测试集下YOLOv3与其它检测算法模型的对比，从mAP和参考时间综合来看，YOLOv3的性能是最高的。

YOLOv3借鉴了残差网络（residual network）的做法，采用含有53个卷积层的Darknet-53的网络结构，使其拥有更快的速度，以下为基于416*416输入的YOLOv3网络图。


## 环境要求

 1. windows
 2. Anaconda (Python 3)
 3. Keras 2.1.5
 4. Tensorflow 1.6.0






## VOC2007 格式数据集制作及训练
打开Anaconda，到 labelImg 路径下，执行以下命令：
> conda install pyqt=5
pyrcc5 -o libs/resources.py resources.qrc
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

        成功安装lambellmg,将训练图片放到JPEGImages下，标注文件保存到Annotations;
        每个图片和标注得到的xml文件，JPEGImages文件夹里面的一个训练图片，对应Annotations里面的一个同名XML文件，一一对应，命名一致;
        数据集制作完成，为了更好的训练结果，可以修改train.py中的迭代次数epochs的值，默认设置为500；修改batch_size 的大小，默认batch_size = 10，此值越大，对GPU显存要求越大，epochs 为训练周期数
训练完成后，训练参数存放在 logs/000 文件夹下，其中，有一个名为 trained_weights_final.h5 即为最后生成的权重文件，将此文件复制到model_data文件夹下，并命名为 yolo.h5 即完成训练。




## 利用数据集进行识别
先修改yolo.py文件中的模型路径，改为自己训练后生成的h5文件路径；
识别图片命令:
>python yolo_video.py --image [image_path]

识别视频命令：
>python yolo_video.py [video_path]

## 识别示例


![image](https://github.com/hitwh613/Perception-group/blob/master/%E5%93%A8%E5%B2%97%E6%95%B0%E6%8D%AE%E8%AE%AD%E7%BB%83%E5%92%8C%E8%AF%86%E5%88%AB/%E8%AF%86%E5%88%AB%E7%A4%BA%E4%BE%8B.jpg)
