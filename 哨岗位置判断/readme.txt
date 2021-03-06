1. 项目介绍
    （1）该项目是对于 RM2020 人工智能挑战赛，哨岗部分测定敌方机器人位姿的代码。可以较为准确的实现对敌方机器人位姿的判断。

2.如何运行
    （1）使用c++环境，并需要安装和配置opencv的环境。

3.文件目录结构：
       toushi文件夹中包含主程序代码，其余均为项目文件。

4.算法原理：
      （1）透视变换：透视变换是将图像从一个视平面投影到另外一个视平面的过程。透视变换本质上空间立体三维变换，是将一个平面通过一个投影矩阵投影到指定平面上。我们依靠透视变换得到位姿信息。
      
      （2）gamma变换：伽玛校正就是对图像的伽玛曲线进行编辑，以对图像进行非线性色调编辑的方法，检出图像信号中的深色部分和浅色部分，并使两者比例增大，从而提高图像对比度效果。我们使用gamma
矫正来去除比赛场地中黄色色块带来的可能影响。

5.相关算法设计：
      （1）展示工程性：该算法的主要解决判定机器人的位姿问题，算法基础是透视变换算法，由于对哨岗摄像头拍摄到的画面直接识别难度过大，情况过于复杂，所以我们希望将视角变化，通过一个相对绝对的视角
来解决位姿问题。所以我们使用了透视变换算法将摄像头拍摄到的画面转换到了俯视平面，得到鸟瞰图效果。由于透视变换只适合于对单个平面进行处理，所以我们将基准点的高度取在与机器人尾部的黄色灯条相同
高度（20cm），这样就可以准确的得到小车光条的实时位姿，由于黄色光条与小车整体朝向是垂直关系，所以进而可以得到小车的位姿信息。
              在具体实现中，透视变换过的图像需要先经过gamma矫正，通过增加对比度的方式，消除场地中障碍物的黄色标记对于识别的影响，然后通过hsv通道提取目标黄色灯条的轮廓，随后通过识别角点，最小二
乘法拟合得到黄色光条（灯条等效成线段）的斜率，进而得到代表小车的斜率K，进而算出朝向角度。
  具体测试效果详见（包含系统框架与流程）：https://youtu.be/4-SoI6jig4M         
       
      （2）方法创新性：
                   优点：算法利用了透视变换的优势，使得机器人特征部分的位姿信息得到了准确的保留，忽略了透视变换在立体空间中的弊端，可以由此为基础，比较准确地计算出车辆的位姿信息。
                   缺点：测试场地要求较高，需要高度契合真实比赛场地，否则提取目标颜色易出现干扰，影响最终效果。
