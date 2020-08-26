#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include "targetver.h"
#include "PNPSolver.h"
#include "GetDistanceOf2linesIn3D.h"
using namespace std;

const int F1 = 2.6;//�����ͷ����
const int F2 = 3.66;

//�ú������ڼ���ĳ�����������
//�������
//pointInImage1���������ͼ1�еĶ�ά����
//p4psolver1��ͼ1�����PNPSolver��
//pointInImage2���������ͼ2�еĶ�ά����
//p4psolver2��ͼ2�����PNPSolver��
//���ش��������������ϵ������
cv::Point3f GetPointInWorld(cv::Point2f pointInImage1, PNPSolver& p4psolver1, cv::Point2f pointInImage2, PNPSolver& p4psolver2)
{
	//��PͶ�䵽�������ϵ1���پ������η���ת�������Oc1P�����ջ��ֱ��Oc1P�ϵ����������꣬ȷ����ֱ�ߵķ���
	cv::Point3f point2find1_CF1 = p4psolver1.ImageFrame2CameraFrame(pointInImage1, F1);//�����P�ڵ�һ̨���������ϵ����
	double Oc1P_x1 = point2find1_CF1.x;
	double Oc1P_y1 = point2find1_CF1.y;
	double Oc1P_z1 = point2find1_CF1.z;
	//�������η�����ת���õ���������ϵ������Oc1P��ֵ
	PNPSolver::CodeRotateByZ(Oc1P_x1, Oc1P_y1, p4psolver1.Theta_W2C.z, Oc1P_x1, Oc1P_y1);
	PNPSolver::CodeRotateByY(Oc1P_x1, Oc1P_z1, p4psolver1.Theta_W2C.y, Oc1P_x1, Oc1P_z1);
	PNPSolver::CodeRotateByX(Oc1P_y1, Oc1P_z1, p4psolver1.Theta_W2C.x, Oc1P_y1, Oc1P_z1);
	//ͨ����֪����ȷ����һ��ֱ��
	cv::Point3f a1(p4psolver1.Position_OcInW.x, p4psolver1.Position_OcInW.y, p4psolver1.Position_OcInW.z);
	cv::Point3f a2(p4psolver1.Position_OcInW.x + Oc1P_x1, p4psolver1.Position_OcInW.y + Oc1P_y1, p4psolver1.Position_OcInW.z + Oc1P_z1);

	//��PͶ�䵽�������ϵ���پ������η���ת�������Oc2P�����ջ��ֱ��Oc2P�ϵ����������꣬ȷ����ֱ�ߵķ���
	cv::Point3f point2find2_CF2 = p4psolver2.ImageFrame2CameraFrame(pointInImage2, F2);//�����P�ڵڶ�̨���������ϵ����
	double Oc2P_x2 = point2find2_CF2.x;
	double Oc2P_y2 = point2find2_CF2.y;
	double Oc2P_z2 = point2find2_CF2.z;
	//�������η�����ת���õ���������ϵ������Oc2P��ֵ
	PNPSolver::CodeRotateByZ(Oc2P_x2, Oc2P_y2, p4psolver2.Theta_W2C.z, Oc2P_x2, Oc2P_y2);
	PNPSolver::CodeRotateByY(Oc2P_x2, Oc2P_z2, p4psolver2.Theta_W2C.y, Oc2P_x2, Oc2P_z2);
	PNPSolver::CodeRotateByX(Oc2P_y2, Oc2P_z2, p4psolver2.Theta_W2C.x, Oc2P_y2, Oc2P_z2);
	//����ȷ��һ��ֱ��
	cv::Point3f b1(p4psolver2.Position_OcInW.x, p4psolver2.Position_OcInW.y, p4psolver2.Position_OcInW.z);
	cv::Point3f b2(p4psolver2.Position_OcInW.x + Oc2P_x2, p4psolver2.Position_OcInW.y + Oc2P_y2, p4psolver2.Position_OcInW.z + Oc2P_z2);

	/*************************���P������**************************/
	//�������ǻ���˹��ڵ�P������ֱ��A1A2��B1B2
	//������ֱ�ߵĽ�����ǵ�P��λ��
	//�����ڴ��ڲ���������ֱ�߲��������غϵģ������˶������
	//�������ֱ������ĵ㣬����P���ڵ�λ���ˡ�

	GetDistanceOf2linesIn3D g;//��ʼ��
	g.SetLineA(a1.x, a1.y, a1.z, a2.x, a2.y, a2.z);//����ֱ��A�ϵ�����������
	g.SetLineB(b1.x, b1.y, b1.z, b2.x, b2.y, b2.z);//����ֱ��B�ϵ�����������
	g.GetDistance();//�������
	double d = g.distance;//��þ���
	//��PonA��PonB�ֱ�Ϊֱ��A��B����ӽ��ĵ㣬���ǵ��е����P������
	double x = (g.PonA_x + g.PonB_x) / 2;
	double y = (g.PonA_y + g.PonB_y) / 2;
	double z = (g.PonA_z + g.PonB_z) / 2;

	return cv::Point3f(x, y, z);
}

//������ͨ������ͼ��λ�ˣ����δ֪��P�Ŀռ����꣨�������꣩
//��ϸԭ����˵����
int main()
{
	//��ʼ������//

	//����ڲ���
	double camD1[9] = {
	1277.23656193959 ,12.26698 ,  1034.9681,
	 0 ,  1283.64158 ,  548.9057,
	 0    ,   0  ,1 };

	double fx = camD1[0];
	double fy = camD1[4];
	double u0 = camD1[2];
	double v0 = camD1[5];

	//��ͷ�������
	double k1 = 0.1498;
	double k2 = 0.7664;
	double p1 = -0.00198;
	double p2 = 0.01145;
	double k3 = -2.3166;

	/********��1��ͼ********/
	PNPSolver p4psolver1;
	//��ʼ���������
	p4psolver1.SetCameraMatrix(fx, fy, u0, v0);
	//���û������
	p4psolver1.SetDistortionCoefficients(k1, k2, p1, p2, k3);

	p4psolver1.Points3D.push_back(cv::Point3f(0, 0, 0));		//P1��ά����ĵ�λ�Ǻ���
	p4psolver1.Points3D.push_back(cv::Point3f(0, 130, 0));		//P2
	p4psolver1.Points3D.push_back(cv::Point3f(135, 0, 0));		//P3
	p4psolver1.Points3D.push_back(cv::Point3f(135, 130, 0));	//P4
	//p4psolver1.Points3D.push_back(cv::Point3f(0, 100, 105));	//P5

	cout << "�������������� = " << endl << p4psolver1.Points3D << endl << endl << endl;

	//���ͼһ�м���������������P������
	//cv::Mat img1 = cv::imread("1.jpg");
	//ͨ��Ŀ��ʶ��ĵ���������
	p4psolver1.Points2D.push_back(cv::Point2f(769, 269));	//P1
	p4psolver1.Points2D.push_back(cv::Point2f(744, 764));	//P2
	p4psolver1.Points2D.push_back(cv::Point2f(1274, 239));	//P3
	p4psolver1.Points2D.push_back(cv::Point2f(1318, 752));	//P4
	//p4psolver1.Points2D.push_back(cv::Point2f(4148, 673));	//P5 ����ʹ���ĸ����P4P�㷨

	cout << "ͼһ������������ = " << endl << p4psolver1.Points2D << endl;

	if (p4psolver1.Solve(PNPSolver::METHOD::CV_P3P) != 0)
		return -1;

	cout << "ͼһ�����λ��" << endl << "Oc����=" << p4psolver1.Position_OcInW << "      �����ת=" << p4psolver1.Theta_W2C << endl;
	cout << endl << endl;

	double camD2[9] = {
			1390.154697753480 ,-45.093318915521 ,  527.009111222550,
					  0 ,  1330.920219359230 ,  127.842764364059,
					  0   ,  0  , 1 };

	fx = camD2[0];
	fy = camD2[4];
	u0 = camD2[2];
	v0 = camD2[5];

	//��ͷ�������
	k1 = -0.20759;
	k2 = -2.09189;
	p1 = 0.0538967;
	p2 = 0.047839;
	k3 = 3.7615594;

	/********��2��ͼ********/
	PNPSolver p4psolver2;
	//��ʼ���������
	p4psolver2.SetCameraMatrix(fx, fy, u0, v0);
	//�������
	p4psolver2.SetDistortionCoefficients(k1, k2, p1, p2, k3);

	p4psolver2.Points3D.push_back(cv::Point3f(0, 0, 0));		//��ά����ĵ�λ�Ǻ���
	p4psolver2.Points3D.push_back(cv::Point3f(0, 130, 0));		//P2
	p4psolver2.Points3D.push_back(cv::Point3f(135, 0, 0));		//P3
	p4psolver2.Points3D.push_back(cv::Point3f(135, 130, 0));	//P4
	//p4psolver2.Points3D.push_back(cv::Point3f(0, 100, 105));	//P5 ����ʹ���ĸ����P4P�㷨

	//���ͼ���м���������������P������
	//cv::Mat img2 = cv::imread("2.jpg");
	p4psolver2.Points2D.push_back(cv::Point2f(389, 126));	//P1
	p4psolver2.Points2D.push_back(cv::Point2f(379, 524));	//P2
	p4psolver2.Points2D.push_back(cv::Point2f(809, 129));	//P3
	p4psolver2.Points2D.push_back(cv::Point2f(797, 532));	//P4
	//p4psolver2.Points2D.push_back(cv::Point2f(3439, 2691));	//P5
	cout << "ͼ�������������� = " << endl << p4psolver2.Points2D << endl;
	if (p4psolver2.Solve(PNPSolver::METHOD::CV_P3P) != 0)
		return -1;
	cout << "ͼ�������λ��" << endl << "Oc����=" << p4psolver2.Position_OcInW << "      �����ת=" << p4psolver2.Theta_W2C << endl;

	cv::Point2f point2find1_IF = cv::Point2f(1020, 493);//�����P��ͼ1������
	cv::Point2f point2find2_IF = cv::Point2f(590, 326);//�����P��ͼ2������

	cv::Point3f p = GetPointInWorld(point2find1_IF, p4psolver1, point2find2_IF, p4psolver2);
	cout << endl << "-------------------------------------------------------------" << endl;
	cout << "���P�������� = (" << p.x << "," << p.y << "," << p.z << ")" << endl;

	//ע��Ϊ�˸���ȷ�ļ�����ռ����꣬���Լ��������P��λ�ã���ȡ���ǵ�����
	return 0;
}