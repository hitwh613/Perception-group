#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include "targetver.h"
#include "PNPSolver.h"
#include "GetDistanceOf2linesIn3D.h"
using namespace std;

const int F1 = 2.6;//相机镜头焦距
const int F2 = 3.66;

//该函数用于计算某点的世界坐标
//输入参数
//pointInImage1：待求点在图1中的二维坐标
//p4psolver1：图1解出的PNPSolver类
//pointInImage2：待求点在图2中的二维坐标
//p4psolver2：图2解出的PNPSolver类
//返回待求点在世界坐标系下坐标
cv::Point3f GetPointInWorld(cv::Point2f pointInImage1, PNPSolver& p4psolver1, cv::Point2f pointInImage2, PNPSolver& p4psolver2)
{
	//将P投射到相机坐标系1，再经过三次反旋转求出向量Oc1P，最终获得直线Oc1P上的两个点坐标，确定了直线的方程
	cv::Point3f point2find1_CF1 = p4psolver1.ImageFrame2CameraFrame(pointInImage1, F1);//待求点P在第一台的相机坐标系坐标
	double Oc1P_x1 = point2find1_CF1.x;
	double Oc1P_y1 = point2find1_CF1.y;
	double Oc1P_z1 = point2find1_CF1.z;
	//进行三次反向旋转，得到世界坐标系中向量Oc1P的值
	PNPSolver::CodeRotateByZ(Oc1P_x1, Oc1P_y1, p4psolver1.Theta_W2C.z, Oc1P_x1, Oc1P_y1);
	PNPSolver::CodeRotateByY(Oc1P_x1, Oc1P_z1, p4psolver1.Theta_W2C.y, Oc1P_x1, Oc1P_z1);
	PNPSolver::CodeRotateByX(Oc1P_y1, Oc1P_z1, p4psolver1.Theta_W2C.x, Oc1P_y1, Oc1P_z1);
	//通过已知两点确定出一条直线
	cv::Point3f a1(p4psolver1.Position_OcInW.x, p4psolver1.Position_OcInW.y, p4psolver1.Position_OcInW.z);
	cv::Point3f a2(p4psolver1.Position_OcInW.x + Oc1P_x1, p4psolver1.Position_OcInW.y + Oc1P_y1, p4psolver1.Position_OcInW.z + Oc1P_z1);

	//将P投射到相机坐标系，再经过三次反旋转求出向量Oc2P，最终获得直线Oc2P上的两个点坐标，确定了直线的方程
	cv::Point3f point2find2_CF2 = p4psolver2.ImageFrame2CameraFrame(pointInImage2, F2);//待求点P在第二台的相机坐标系坐标
	double Oc2P_x2 = point2find2_CF2.x;
	double Oc2P_y2 = point2find2_CF2.y;
	double Oc2P_z2 = point2find2_CF2.z;
	//进行三次反向旋转，得到世界坐标系中向量Oc2P的值
	PNPSolver::CodeRotateByZ(Oc2P_x2, Oc2P_y2, p4psolver2.Theta_W2C.z, Oc2P_x2, Oc2P_y2);
	PNPSolver::CodeRotateByY(Oc2P_x2, Oc2P_z2, p4psolver2.Theta_W2C.y, Oc2P_x2, Oc2P_z2);
	PNPSolver::CodeRotateByX(Oc2P_y2, Oc2P_z2, p4psolver2.Theta_W2C.x, Oc2P_y2, Oc2P_z2);
	//两点确定一条直线
	cv::Point3f b1(p4psolver2.Position_OcInW.x, p4psolver2.Position_OcInW.y, p4psolver2.Position_OcInW.z);
	cv::Point3f b2(p4psolver2.Position_OcInW.x + Oc2P_x2, p4psolver2.Position_OcInW.y + Oc2P_y2, p4psolver2.Position_OcInW.z + Oc2P_z2);

	/*************************求出P的坐标**************************/
	//现在我们获得了关于点P的两条直线A1A2与B1B2
	//于是两直线的交点便是点P的位置
	//但由于存在测量误差，两条直线不可能是重合的，于是退而求其次
	//求出两条直线最近的点，就是P所在的位置了。

	GetDistanceOf2linesIn3D g;//初始化
	g.SetLineA(a1.x, a1.y, a1.z, a2.x, a2.y, a2.z);//输入直线A上的两个点坐标
	g.SetLineB(b1.x, b1.y, b1.z, b2.x, b2.y, b2.z);//输入直线B上的两个点坐标
	g.GetDistance();//计算距离
	double d = g.distance;//获得距离
	//点PonA与PonB分别为直线A、B上最接近的点，他们的中点就是P的坐标
	double x = (g.PonA_x + g.PonB_x) / 2;
	double y = (g.PonA_y + g.PonB_y) / 2;
	double z = (g.PonA_z + g.PonB_z) / 2;

	return cv::Point3f(x, y, z);
}

//本程序通过两副图的位姿，求出未知点P的空间坐标（世界坐标）
//详细原理与说明：
int main()
{
	//初始化参数//

	//相机内参数
	double camD1[9] = {
	1277.23656193959 ,12.26698 ,  1034.9681,
	 0 ,  1283.64158 ,  548.9057,
	 0    ,   0  ,1 };

	double fx = camD1[0];
	double fy = camD1[4];
	double u0 = camD1[2];
	double v0 = camD1[5];

	//镜头畸变参数
	double k1 = 0.1498;
	double k2 = 0.7664;
	double p1 = -0.00198;
	double p2 = 0.01145;
	double k3 = -2.3166;

	/********第1幅图********/
	PNPSolver p4psolver1;
	//初始化相机参数
	p4psolver1.SetCameraMatrix(fx, fy, u0, v0);
	//设置畸变参数
	p4psolver1.SetDistortionCoefficients(k1, k2, p1, p2, k3);

	p4psolver1.Points3D.push_back(cv::Point3f(0, 0, 0));		//P1三维坐标的单位是毫米
	p4psolver1.Points3D.push_back(cv::Point3f(0, 130, 0));		//P2
	p4psolver1.Points3D.push_back(cv::Point3f(135, 0, 0));		//P3
	p4psolver1.Points3D.push_back(cv::Point3f(135, 130, 0));	//P4
	//p4psolver1.Points3D.push_back(cv::Point3f(0, 100, 105));	//P5

	cout << "特征点世界坐标 = " << endl << p4psolver1.Points3D << endl << endl << endl;

	//求出图一中几个特征点与待求点P的坐标
	//cv::Mat img1 = cv::imread("1.jpg");
	//通过目标识别的的像素坐标
	p4psolver1.Points2D.push_back(cv::Point2f(769, 269));	//P1
	p4psolver1.Points2D.push_back(cv::Point2f(744, 764));	//P2
	p4psolver1.Points2D.push_back(cv::Point2f(1274, 239));	//P3
	p4psolver1.Points2D.push_back(cv::Point2f(1318, 752));	//P4
	//p4psolver1.Points2D.push_back(cv::Point2f(4148, 673));	//P5 仅仅使用四个点的P4P算法

	cout << "图一中特征点坐标 = " << endl << p4psolver1.Points2D << endl;

	if (p4psolver1.Solve(PNPSolver::METHOD::CV_P3P) != 0)
		return -1;

	cout << "图一中相机位姿" << endl << "Oc坐标=" << p4psolver1.Position_OcInW << "      相机旋转=" << p4psolver1.Theta_W2C << endl;
	cout << endl << endl;

	double camD2[9] = {
			1390.154697753480 ,-45.093318915521 ,  527.009111222550,
					  0 ,  1330.920219359230 ,  127.842764364059,
					  0   ,  0  , 1 };

	fx = camD2[0];
	fy = camD2[4];
	u0 = camD2[2];
	v0 = camD2[5];

	//镜头畸变参数
	k1 = -0.20759;
	k2 = -2.09189;
	p1 = 0.0538967;
	p2 = 0.047839;
	k3 = 3.7615594;

	/********第2幅图********/
	PNPSolver p4psolver2;
	//初始化相机参数
	p4psolver2.SetCameraMatrix(fx, fy, u0, v0);
	//畸变参数
	p4psolver2.SetDistortionCoefficients(k1, k2, p1, p2, k3);

	p4psolver2.Points3D.push_back(cv::Point3f(0, 0, 0));		//三维坐标的单位是毫米
	p4psolver2.Points3D.push_back(cv::Point3f(0, 130, 0));		//P2
	p4psolver2.Points3D.push_back(cv::Point3f(135, 0, 0));		//P3
	p4psolver2.Points3D.push_back(cv::Point3f(135, 130, 0));	//P4
	//p4psolver2.Points3D.push_back(cv::Point3f(0, 100, 105));	//P5 仅仅使用四个点的P4P算法

	//求出图二中几个特征点与待求点P的坐标
	//cv::Mat img2 = cv::imread("2.jpg");
	p4psolver2.Points2D.push_back(cv::Point2f(389, 126));	//P1
	p4psolver2.Points2D.push_back(cv::Point2f(379, 524));	//P2
	p4psolver2.Points2D.push_back(cv::Point2f(809, 129));	//P3
	p4psolver2.Points2D.push_back(cv::Point2f(797, 532));	//P4
	//p4psolver2.Points2D.push_back(cv::Point2f(3439, 2691));	//P5
	cout << "图二中特征点坐标 = " << endl << p4psolver2.Points2D << endl;
	if (p4psolver2.Solve(PNPSolver::METHOD::CV_P3P) != 0)
		return -1;
	cout << "图二中相机位姿" << endl << "Oc坐标=" << p4psolver2.Position_OcInW << "      相机旋转=" << p4psolver2.Theta_W2C << endl;

	cv::Point2f point2find1_IF = cv::Point2f(1020, 493);//待求点P在图1中坐标
	cv::Point2f point2find2_IF = cv::Point2f(590, 326);//待求点P在图2中坐标

	cv::Point3f p = GetPointInWorld(point2find1_IF, p4psolver1, point2find2_IF, p4psolver2);
	cout << endl << "-------------------------------------------------------------" << endl;
	cout << "解得P世界坐标 = (" << p.x << "," << p.y << "," << p.z << ")" << endl;

	//注：为了更精确的计算出空间坐标，可以计算出多组P的位置，并取它们的重心
	return 0;
}