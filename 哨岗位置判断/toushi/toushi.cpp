

#include <stdio.h>
#include <opencv/highgui.h>  
#include <time.h>  
#include <opencv2/opencv.hpp>  
#include <opencv/cv.h>  
#include <iostream> 

using namespace std;
using namespace cv;


Mat Result, gray_Result;
Mat tomasi_res;

//存储图像中的最大值和最小值，计算角点使用
double tomasi_min;
double tomasi_max;

//计算百分比,拖动TrackBar，计算角点使用
int current_value = 43;
int max_value = 50;

const char* output_title = "custom tomasi corner title";

Mat perspectiveTransformation(Mat& src, Mat& resultimage)//该函数用于透视变换，变换平面高度（基准边的高度）与机器人尾部黄色灯条高度一致
{  
	vector<Point2f>src_coners(4);     //4个原图像上的透视变换的基准点的坐标
	src_coners[0] = Point2f(811, 116);
	src_coners[1] = Point2f(1890, 119);
	src_coners[2] = Point2f(163, 430);
	src_coners[3] = Point2f(1677, 458);

	//对标记点以红点作为标记
	circle(src, src_coners[0], 3, Scalar(0, 0, 255), 3, 8);
	circle(src, src_coners[1], 3, Scalar(0, 0, 255), 3, 8);
	circle(src, src_coners[2], 3, Scalar(0, 0, 255), 3, 8);
	circle(src, src_coners[3], 3, Scalar(0, 0, 255), 3, 8);
	vector<Point2f>dst_coners(4);  //变换后四个基准点对应的新坐标
	dst_coners[0] = Point2f(0, 0);
	dst_coners[1] = Point2f(1920, 0);
	dst_coners[2] = Point2f(0, 1080);
	dst_coners[3] = Point2f(1920, 1080);
	Mat warpMatrix = getPerspectiveTransform(src_coners, dst_coners);//得到所需的透视变换矩阵
	warpPerspective(src, resultimage, warpMatrix, resultimage.size(), INTER_LINEAR, BORDER_CONSTANT);//对图像进行透视变换
	return resultimage;
}


Mat sharpen2D(const Mat& image, Mat& result) // 该函数是基于filter2D()的的图像锐化函数，用于帮助寻找角点
{
	 // 首先构造一个内核
	Mat kernel(3, 3, CV_32F, Scalar(0));
	 // 对 对应内核进行赋值
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	/// 对图像进行滤波操作
	filter2D(image, result, image.depth(), kernel);
	return result;
}

	
	

 Mat MyGammaCorrection(Mat& src, Mat& dst, float fGamma) 
	 // 该函数为Gamma变换函数，用于提高透视变换后图像的对比度，去除可能存在的干扰
	 // 系数fGamma设定为10，较暗的区域被压缩的更暗，使得障碍物上的黄色标识不会对黄色指示灯的识别造成影响
{
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:   //灰度图的情况
	{

		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++) 

			*it = lut[(*it)];

		break;
	}
	case 3:  //彩色图的情况
	{

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
	return dst;//返回经过Gamma变换后的图像，方便颜色识别
}


Mat filteredRed(const Mat& inputImage, Mat& resultGray, Mat& resultColor) 
//该函数为颜色识别函数，用于提取黄色灯条的轮廓
//使用HSV颜色系统，事先已对目标颜色hsv范围进行过测定。
{
	Mat hsvImage;
	cvtColor(inputImage, hsvImage, CV_BGR2HSV);
	resultGray = Mat(hsvImage.rows, hsvImage.cols, CV_8U, cv::Scalar(255));
	resultColor = Mat(hsvImage.rows, hsvImage.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	double H = 0.0, S = 0.0, V = 0.0; //定义HSV三项，H(色相),S(饱和度),V(色调) 
	for (int i = 0; i < hsvImage.rows; i++)//遍历整个图像，寻找符合要求的像素点
	{
		for (int j = 0; j < hsvImage.cols; j++)
		{
			H = hsvImage.at<Vec3b>(i, j)[0];
			S = hsvImage.at<Vec3b>(i, j)[1];
			V = hsvImage.at<Vec3b>(i, j)[2];

			if ((S >= 30 && S < 180))
			{
				if ((H >= 30 && H < 80) && V >= 225)//写入测定HSV的数值范围
				{
					resultGray.at<uchar>(i, j) = 0;
					resultColor.at<Vec3b>(i, j)[0] = inputImage.at<Vec3b>(i, j)[0];
					resultColor.at<Vec3b>(i, j)[1] = inputImage.at<Vec3b>(i, j)[1];
					resultColor.at<Vec3b>(i, j)[2] = inputImage.at<Vec3b>(i, j)[2];
				}

			}
		}
	}

	return resultGray;//返回灯条的轮廓
}




void custom_tomasi_corner1(int, void*) //该函数用于寻找灯条轮廓上的角点，并返回图像方便调试
{

	Mat resultImg = Result.clone();
	float t = tomasi_min + (((double)current_value) / max_value) * (tomasi_max - tomasi_min);
	for (int col = 0; col < Result.cols; col++) //遍历所有像素点，寻找角点
	{
		for (int row = 0; row < Result.rows; row++)
		{
			float v = tomasi_res.at<float>(row, col);
			if (v > t)
			{
				circle(resultImg, Point(col, row), 2, Scalar(0, 0, 255), 2, 8, 0);//角点用红点标出
		
			}
		}


	}
	imshow(output_title, resultImg);//输出具有标注角点的图像，方便调试

}


float custom_tomasi_corner2(int, void*)//该函数与上一个函数内容相似，在找到角点的基础上使用最小二乘法进行拟合，算出代表黄色灯条的等效直线斜率
{
	float point_x[100];
	float point_y[100];
	//定义数组用于记录角点的坐标
	int a = 0;
	Mat resultImg = Result.clone();
	float t = tomasi_min + (((double)current_value) / max_value) * (tomasi_max - tomasi_min);
	for (int row = 0; row < Result.rows; row++)
	{
		for (int col = 0; col < Result.cols; col++)
		{
			float v = tomasi_res.at<float>(row, col);
			if (v > t)
			{
				circle(resultImg, Point(col, row), 2, Scalar(0, 0, 255), 2, 8, 0);
				//以上仍是寻找角点，然后记录角点坐标
				point_x[a] = col;
				point_y[a] = 770 - row;//像素坐标系与一般使用的数学坐标系y轴方向相反，所以这里需要调整
				a = a + 1;
			}
		}


	}
	float A = 0.0;
	float B = 0.0;
	float C = 0.0;
	float D = 0.0;
	float E = 0.0;
	float F = 0.0;
	//使用最小二乘法拟合曲线，先定义相关量
	for (int i = 0; i < a; i++)//根据角点坐标与数学关系式，计算相关量
	{
		A += (point_x[i]) * (point_x[i]);
		B += point_x[i];
		C += (point_x[i]) * (point_y[i]);
		D += point_y[i];
	}
	
	// 计算斜率k
	float k, temp = 0;
	if (temp = (a * A - B * B))// 判断分母不为0
	{
		k = (a * C - B * D) / temp;
	}
	else
	{
		k = 1;
	}
	return -1 / k;//等效车辆朝向的直线斜率与灯条直线垂直，所以输出-1/k
}


void main()
{
	Mat src = imread("1.jpg"); //读取图像
	Mat resultimage;
	Mat result;
	Mat result1;
	float fGamma = 10;//定义Gamma变换系数，为了有效增加对比度，定为10
	const int pai = 3.14159265358979323;//定义π，用于将角度、从弧度制转变为角度制
	int blocksize = 3;//；领域尺寸，计算角点使用
	int ksize = 3;// 算子的孔径参数，此处取3，计算角点使用
	float k;//斜率
	float alf1;//可能角度1
	float alf2;//可能角度2
	resultimage = perspectiveTransformation(src, resultimage);//透视变换
	result= MyGammaCorrection(resultimage, result1, fGamma);//gamma变化
	resize(resultimage, resultimage, Size(765, 577.5), 0, 0, INTER_CUBIC);//改变图像大小，比例应该与透视变换的地块比例相同
	resize(result, result, Size(765, 577.5), 0, 0, INTER_CUBIC);
	Mat greysrc;
	Mat resultsrc;
	Mat resultgray;
	resultgray = filteredRed(result, greysrc, resultsrc);//提取目标色块
	result = sharpen2D(resultgray, result);//锐化图像
	//imshow("src", src);
	imshow("trans", resultimage);
	waitKey(0);
	string path;
	path = "C:\\Users\\evra\\source\\repos\\toushi\\toushi\\2.jpg";//储存图片
	imwrite(path, result);
	//imshow("trans", result);
	//waitKey (0);
	Result = imread("C:\\Users\\evra\\source\\repos\\toushi\\toushi\\2.jpg");
	cvtColor(Result, gray_Result, COLOR_BGR2GRAY);
	tomasi_res = Mat::zeros(Result.size(), CV_32FC1);
	imshow("input title", Result);

	cornerMinEigenVal(gray_Result, tomasi_res, blocksize, ksize, BORDER_DEFAULT);// 计算角点

	minMaxLoc(tomasi_res, &tomasi_min, &tomasi_max, 0, 0, Mat());
	namedWindow(output_title, CV_WINDOW_AUTOSIZE);

	createTrackbar("tomasi-bar", output_title, &current_value, max_value, custom_tomasi_corner1);
	custom_tomasi_corner1(0, 0);
	k = custom_tomasi_corner2(0, 0);
	cout <<"斜率="<< k <<"\n" ;
	alf1 = atan(k) * 180 / pai;//反三角函数得出角度，并从弧度制转角度制
	if (alf1 < 0)//求出可能的两个角度
	{
		alf2 = alf1 + 180;
	}
	else
	{
		alf2 = alf1 - 180;
	}
	cout <<"角度1="<< setprecision(2) << alf1 << "\n";
	cout <<"角度2="<< setprecision(2) << alf2 << "\n";
	waitKey(0);
}
	
