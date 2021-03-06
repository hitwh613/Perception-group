﻿ 1.项目介绍
（1）该项目是对于 RM2020 人工智能挑战赛，哨岗部分定位敌方机器人的部分，可以较为准确的实现对敌方机器人的定位。本文将展示位姿估计的一种应用，即通过两个单目相机完成对物体距离的测量和定位。简单来说，本文的工作就是利用两幅图，在已知P1、P2、P3、P4四点世界坐标的情况下，计算出P5的世界坐标。本文仅仅简要叙述原理及相应函数说明，更详细的原理及演示图和函数说明请看PDF原理和函数注释。

2.运行环境
（1）代码运行环境：Visualstudio2019，Python3.7，Opencv4.1.1

3.文件目录结构
（1）Dingwei文件夹中包括定位核心代码Dingwei.cpp和说明文件readme.txt以及图表原理及结果验证Dingwei.pdf

4.算法原理
（1）根据两条直线确定一个点的原理，在二维平面中只要知道两条相交直线的方程，就可以解出它们的交点坐标，我们根据P1、P2、P3、P4四点的空间坐标，利用PnP原理可以估计出两次拍照的相机位姿Oc1与Oc2，也就知道了相机的坐标Oc1与Oc2。那么将Oc1与P，Oc2与P联成直线，则可以获得两条直线方程，组成方程组求解得到它们的交点，即为待求点P的坐标。

（2）PnP求解算法的主要内容是通过多对3D与2D匹配点，在已知相机内参的情况下，利用最小化重投影误差来求解相机外参的算法。建立以场地边缘为原点的X-Y-Z世界坐标系，标定所选已知点的世界坐标，将标定好的摄像头内参矩阵和相机畸变系数代入PnP解算中，利用Effiencient-EPNP方法，计算出旋转矩阵和平移矩阵，即相机坐标系的变换矩阵。

5.算法设计工程
（1）求出P点的相机坐标系坐标。
  通过P点的像素坐标，利用相机内外参数转换公式，将P点的像素坐标转换为相机坐标系坐标。同时通过PNP处理P1-P4四个点获得相机的坐标Oc1与Oc2。

（2）代码使用了封装好的解PNP问题的类函数PNPSolver来解决PNP问题，输入初始化相机参数和畸变参数，再分别输入三维坐标和和像素坐标，即可利用PNP解出相机Oc的坐标。

（3）求出P点在世界坐标系坐标
  上述我们已经得到了P点的相机坐标系坐标，由PNP原理可知，使用到解PNP求位姿时得到的三个欧拉角Seta-x Seta-y Seta-z，将相机坐标系C按照z轴、y轴、x轴的顺序旋转以上角度后将与世界坐标系W完全平行，Pc显然是跟着坐标系旋转的，其在世界系W中的位置会随着改变。为了抵消旋转对P点的影响，保证C系旋转后P点依然保持在世界坐标系W原本的位置，需要对Pc进行三次反向旋转，旋转后得到点Pc在相机坐标系C中新的坐标值记为Pc'，Pc'的值等于世界坐标系中向量OP的值。那么Pc'的值+ Oc的世界坐标值=P点的世界坐标Pw

（4）最后，根据POc1和POc2的两条直线，计算出P点的世界坐标
  然而在现实中，由于误差的存在，A与B相交的可能性几乎不存在，因此在计算时，应该求他们之间的最近点坐标。

（5）代码中将求空间内两条直线的最近距离以及最近点的坐标封装成了GetDistanceOf2linesIn3D类，输入两条直线上各两点坐标，即可求出距离和最近点坐标。

6.算法分析
（1）缺点： 两张拍照位置造成的误差，理论上两次拍照位置相互垂直时，最后计算出来的P点世界坐标的误差最小。 当两个摄像头中的一个完全丢失识别物体的视野时，会对测量产生一定影响 
（2）优点：改进的PnP算法的主要作用在于将2个摄像头利用两条直线确定空间交点的原理测定世界坐标系中小车，降低了单一的单目摄像头进行测距和位姿识别的误差，使得PnP算法能够在实时的哨岗系统中提供精确的位置坐标
