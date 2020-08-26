include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"
#include "omp.h"
#include "math.h"
#include "fstream"
#include "stdafx.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"


using namespace cv;
using namespace std;


#define T_ANGLE_THRE 10
#define T_SIZE_THRE 5

void brightAdjust(Mat src, Mat dst, double dContrast, double dBright); //亮度调节函数
void getDiffImage(Mat src1, Mat src2, Mat dst, int nThre); //二值化
vector<RotatedRect> armorDetect(vector<RotatedRect> vEllipse); //检测装甲
void drawBoard(RotatedRect board, Mat img); //标记装甲
void codeRotateByZ(double x, double y, double thetaz, double& outx, double& outy);
void codeRotateByY(double x, double z, double thetay, double& outx, double& outz);
void codeRotateByX(double y, double z, double thetax, double& outy, double& outz);
cv::Point3f RotateByVector(double old_x, double old_y, double old_z, double vx, double vy, double vz, double theta);

int main()
{
    VideoCapture cap(0);
    Mat frame;
    Size imgSize;
    RotatedRect s;   //定义旋转矩形
    vector<RotatedRect> vEllipse; //定以旋转矩形的向量，用于存储发现的目标区域
    vector<RotatedRect> vRlt;
    vector<RotatedRect> vArmor;
    bool bFlag = false;
    vector<vector<Point> > contour;
    cap >> frame;
    imgSize = frame.size();

    Mat rawImg = Mat(imgSize, CV_8UC3);
    Mat grayImage = Mat(imgSize, CV_8UC1);
    Mat rImage = Mat(imgSize, CV_8UC1);
    Mat gImage = Mat(imgSize, CV_8UC1);
    Mat bImage = Mat(imgSize, CV_8UC1);
    Mat binary = Mat(imgSize, CV_8UC1);
    Mat rlt = Mat(imgSize, CV_8UC1);
    namedWindow("raw");
    while (1)
    {
        if (cap.read(frame))
        {
            brightAdjust(frame, rawImg, 1, -120);  //每个像素每个通道的值都减去120
            Mat bgr[3];
            split(rawImg, bgr); //将三个通道的像素值分离
            bImage = bgr[0];
            gImage = bgr[1];
            rImage = bgr[2];
            //如果像素B值-G值大于25，则返回的二值图像的值为255，否则为0
            getDiffImage(bImage, gImage, binary, 25);
            dilate(binary, grayImage, Mat(), Point(-1, -1), 3);   //图像膨胀
            erode(grayImage, rlt, Mat(), Point(-1, -1), 1);  //图像腐蚀，先膨胀在腐蚀属于闭运算
            findContours(rlt, contour, RETR_CCOMP, CHAIN_APPROX_SIMPLE); //在二值图像中寻找轮廓
            for (int i = 0; i < contour.size(); i++)
            {
                if (contour[i].size() > 10)  //判断当前轮廓是否大于10个像素点
                {
                    bFlag = true;   //如果大于10个，则检测到目标区域
                  //拟合目标区域成为椭圆，返回一个旋转矩形（中心、角度、尺寸）
                    s = fitEllipse(contour[i]);
                    for (int nI = 0; nI < 5; nI++)
                    {
                        for (int nJ = 0; nJ < 5; nJ++)  //遍历以旋转矩形中心点为中心的5*5的像素块
                        {
                            if (s.center.y - 2 + nJ > 0 && s.center.y - 2 + nJ < 480 && s.center.x - 2 + nI > 0 && s.center.x - 2 + nI < 640)  //判断该像素是否在有效的位置
                            {
                                Vec3b v3b = frame.at<Vec3b>((int)(s.center.y - 2 + nJ), (int)(s.center.x - 2 + nI)); //获取遍历点点像素值
                               //判断中心点是否接近黑色
                                if (v3b[0] < 50 || v3b[1] < 50 || v3b[2] < 50)
                                    bFlag = false;
                            }
                        }
                    }
                    if (bFlag)
                    {
                        vEllipse.push_back(s); //将发现的目标保存
                    }
                }

            }
            //调用子程序，在输入的LED所在旋转矩形的vector中找出装甲的位置，并包装成旋转矩形，存入vector并返回
            vRlt = armorDetect(vEllipse);
            for (unsigned int nI = 0; nI < vRlt.size(); nI++) //在当前图像中标出装甲的位置
            drawBoard(vRlt[nI], frame);           
            imshow("source", frame);
            if (waitKey(30) == 27)
            {
                break;
            }
            vEllipse.clear();
            vRlt.clear();
            vArmor.clear();
        }
        else
        {
            break;
        }
    }
    cap.release();
    return 0;

}

void brightAdjust(Mat src, Mat dst, double dContrast, double dBright)
{
    int nVal;
    omp_set_num_threads(8);
#pragma omp parallel for

    for (int nI = 0; nI < src.rows; nI++)
    {
        Vec3b* p1 = src.ptr<Vec3b>(nI);
        Vec3b* p2 = dst.ptr<Vec3b>(nI);
        for (int nJ = 0; nJ < src.cols; nJ++)
        {
            for (int nK = 0; nK < 3; nK++)
            {
                //每个像素的每个通道的值都进行线性变换
                nVal = (int)(dContrast * p1[nJ][nK] + dBright);
                if (nVal < 0)
                    nVal = 0;
                if (nVal > 255)
                    nVal = 255;
                p2[nJ][nK] = nVal;
            }
        }
    }
}

void getDiffImage(Mat src1, Mat src2, Mat dst, int nThre)
{
    omp_set_num_threads(8);
#pragma omp parallel for

    for (int nI = 0; nI < src1.rows; nI++)
    {
        uchar* pchar1 = src1.ptr<uchar>(nI);
        uchar* pchar2 = src2.ptr<uchar>(nI);
        uchar* pchar3 = dst.ptr<uchar>(nI);
        for (int nJ = 0; nJ < src1.cols; nJ++)
        {
            if (pchar1[nJ] - pchar2[nJ] > nThre) //
            {
                pchar3[nJ] = 255;
            }
            else
            {
                pchar3[nJ] = 0;
            }
        }
    }
}

vector<RotatedRect> armorDetect(vector<RotatedRect> vEllipse)
{
    vector<RotatedRect> vRlt;
    RotatedRect armor; //定义装甲区域的旋转矩形
    int nL, nW;
    double dAngle;
    vRlt.clear();
    if (vEllipse.size() > 2) //如果检测到的旋转矩形个数小于2，则直接返回
        return vRlt;
    for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++) //求任意两个旋转矩形的夹角
    {
        for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
        {
            dAngle = abs(vEllipse[nI].angle - vEllipse[nJ].angle);
            while (dAngle > 180)
                dAngle -= 180;
            //判断这两个旋转矩形是否是一个装甲的两个LED等条
            if ((dAngle < T_ANGLE_THRE || 180 - dAngle < T_ANGLE_THRE) && abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) < (vEllipse[nI].size.height + vEllipse[nJ].size.height) / T_SIZE_THRE && abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) < (vEllipse[nI].size.width + vEllipse[nJ].size.width) / T_SIZE_THRE)
            {
                armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2; //装甲中心的x坐标 
                armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2; //装甲中心的y坐标
                armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;   //装甲所在旋转矩形的旋转角度
                if (180 - dAngle < T_ANGLE_THRE)
                    armor.angle += 90;
                nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2; //装甲的高度
                nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x) * (vEllipse[nI].center.x - vEllipse[nJ].center.x) + (vEllipse[nI].center.y - vEllipse[nJ].center.y) * (vEllipse[nI].center.y - vEllipse[nJ].center.y)); //装甲的宽度等于两侧LED所在旋转矩形中心坐标的距离
                if (nL <= nW)
                {
                    armor.size.height = nL;
                    armor.size.width = nW;
                }
                else
                {
                    armor.size.height = nW;
                    armor.size.width = nL;
                }
                vRlt.push_back(armor); //将找出的装甲的旋转矩形保存到vector
            }
        }
    }
    return vRlt;
}

void drawBoard(RotatedRect board, Mat img)
{
    Point2f pt[4];
    int i;
    for (i = 0; i < 4; i++)
    {
        pt[i].x = 0;
        pt[i].y = 0;
    }
    board.points(pt); //计算顶点 
    line(img, pt[0], pt[1], CV_RGB(0, 0, 255), 2, 8, 0);
    line(img, pt[1], pt[2], CV_RGB(0, 0, 255), 2, 8, 0);
    line(img, pt[2], pt[3], CV_RGB(0, 0, 255), 2, 8, 0);
    line(img, pt[3], pt[0], CV_RGB(0, 0, 255), 2, 8, 0);
    //初始化相机参数
    double camD[9] = { 1.043688350990551e+03,0,0,0,1.055250864906758e+03,0,5.367960374791403e+02,4.569356169283431e+02,1 };
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_64FC1, camD);
    //畸变参数
    double distCoeffD[5] = { -0.330966274919707,0.054136900398817,0,0,0 };
    cv::Mat distortion_coefficients = cv::Mat(5, 1, CV_64FC1, distCoeffD);
    //特征点世界坐标
    vector<cv::Point3f> Points3D;
    Points3D.push_back(cv::Point3f(500, 460, 0));//P1 单位：毫米
    Points3D.push_back(cv::Point3f(500, -460, 0));//P2
    Points3D.push_back(cv::Point3f(-500, -460, 0));//P3           
    Points3D.push_back(cv::Point3f(-500, 460, 0));//P4
    //特征点像素坐标
    vector<cv::Point2f> Points2D;
    Points2D.push_back(cv::Point2f(pt[0]));	//P1 单位：毫米
    Points2D.push_back(cv::Point2f(pt[1]));	//P2
    Points2D.push_back(cv::Point2f(pt[2]));	//P3           
    Points2D.push_back(cv::Point2f(pt[3]));	//P4
    //初始化输出矩阵
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);
    solvePnP(Points3D, vRlt, camera_matrix, distortion_coefficients, rvec, tvec, false, SOLVEPNP_P3P);
    //旋转向量变旋转矩阵
    //提取旋转矩阵
    double rm[9];
    cv::Mat rotM(3, 3, CV_64FC1, rm);
    Rodrigues(rvec, rotM);
    double r11 = rotM.ptr<double>(0)[0];
    double r12 = rotM.ptr<double>(0)[1];
    double r13 = rotM.ptr<double>(0)[2];
    double r21 = rotM.ptr<double>(1)[0];
    double r22 = rotM.ptr<double>(1)[1];
    double r23 = rotM.ptr<double>(1)[2];
    double r31 = rotM.ptr<double>(2)[0];
    double r32 = rotM.ptr<double>(2)[1];
    double r33 = rotM.ptr<double>(2)[2];
    //计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
    //旋转顺序为z、y、x
    double thetaz = atan2(r21, r11) / CV_PI * 180;
    double thetay = atan2(-1 * r31, sqrt(r32 * r32 + r33 * r33)) / CV_PI * 180;
    double thetax = atan2(r32, r33) / CV_PI * 180;
    cout << "相机的三轴旋转角：" << -1 * thetax << ", " << -1 * thetay << ", " << -1 * thetaz << endl;
    //提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
    double tx = tvec.ptr<double>(0)[0];
    double ty = tvec.ptr<double>(0)[1];
    double tz = tvec.ptr<double>(0)[2];
    cout << "物体的世界坐标：" << tx << ", " << ty << ", " << tz << endl;
    }
}


void codeRotateByZ(double x, double y, double thetaz, double& outx, double& outy)
{
    double x1 = x;//将变量拷贝一次，保证&x == &outx这种情况下也能计算正确
    double y1 = y;
    double rz = thetaz * CV_PI / 180;
    outx = cos(rz) * x1 - sin(rz) * y1;
    outy = sin(rz) * x1 + cos(rz) * y1;
}



//将空间点绕Y轴旋转
//输入参数 x z为空间点原始x z坐标
//thetay为空间点绕Y轴旋转多少度，角度制范围在-180到180
//outx outz为旋转后的结果坐标

void codeRotateByY(double x, double z, double thetay, double& outx, double& outz)

{
    double x1 = x;
    double z1 = z;
    double ry = thetay * CV_PI / 180;
    outx = cos(ry) * x1 + sin(ry) * z1;
    outz = cos(ry) * z1 - sin(ry) * x1;
}



//将空间点绕X轴旋转
//输入参数 y z为空间点原始y z坐标
//thetax为空间点绕X轴旋转多少度，角度制，范围在-180到180
//outy outz为旋转后的结果坐标

void codeRotateByX(double y, double z, double thetax, double& outy, double& outz)

{
    double y1 = y;//将变量拷贝一次，保证&y == &y这种情况下也能计算正确
    double z1 = z;
    double rx = thetax * CV_PI / 180;
    outy = cos(rx) * y1 - sin(rx) * z1;
    outz = cos(rx) * z1 + sin(rx) * y1;
}





//点绕任意向量旋转，右手系
//输入参数old_x，old_y，old_z为旋转前空间点的坐标
//vx，vy，vz为旋转轴向量
//theta为旋转角度角度制，范围在-180到180
//返回值为旋转后坐标点
cv::Point3f RotateByVector(double old_x, double old_y, double old_z, double vx, double vy, double vz, double theta)
{
    double r = theta * CV_PI / 180;
    double c = cos(r);
    double s = sin(r);
    double new_x = (vx * vx * (1 - c) + c) * old_x + (vx * vy * (1 - c) - vz * s) * old_y + (vx * vz * (1 - c) + vy * s) * old_z;
    double new_y = (vy * vx * (1 - c) + vz * s) * old_x + (vy * vy * (1 - c) + c) * old_y + (vy * vz * (1 - c) - vx * s) * old_z;
    double new_z = (vx * vz * (1 - c) - vy * s) * old_x + (vy * vz * (1 - c) + vx * s) * old_y + (vz * vz * (1 - c) + c) * old_z;
    return cv::Point3f(new_x, new_y, new_z);

}
