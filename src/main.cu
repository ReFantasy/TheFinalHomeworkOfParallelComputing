#include <iostream>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include <map>

constexpr int ET = 1;
constexpr double PI = 3.1415926;
using namespace std;
using Point = Vector3d;

cv::Mat ViewGenerate(Vector3d U, Vector3d F, double alpha, double beta, int M, int N);
std::map<PlaneIndex, cv::Mat> images;

float scale(int i, int n)
{
	return ((float)i) / (n - 1);
}



int main()
{
	Vector3d U( 0,0,1 );
	Vector3d F( 1,0,0 );

	
	// 读取图片
	cv::Mat image_up = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\up.jpg");
	cv::Mat image_bottom = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\bottom.jpg");
	cv::Mat image_front = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\front.jpg");
	cv::Mat image_back = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\back.jpg");
	cv::Mat image_left = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\left.jpg");
	cv::Mat image_right = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\right.jpg");

	
	images[PlaneIndex::UP] = image_up;
	images[PlaneIndex::BOTTOM] = image_bottom;
	images[PlaneIndex::FRONT] = image_front;
	images[PlaneIndex::BACK] = image_back;
	images[PlaneIndex::LEFT] = image_left;
	images[PlaneIndex::RIGHT] = image_right;

	auto img = ViewGenerate(U, F, 60 * PI / 180.0, 60 * PI / 180.0, 1024, 1024);
	cv::imshow("result", img);
	cv::waitKey();
	


	
#ifdef _WIN32
	system("pause");
#endif
    return 0;
}


cv::Mat ViewGenerate(Vector3d U, Vector3d F, double alpha, double beta, int M, int N)
{
	// 计算 T 点坐标
	Point T = F.normalized()*ET;
	//cout << "T:" << endl << T << endl;

	// 垂直于平面ETU的法向量
	auto VTQ = F.cross(U);

	// 图像宽和高
	double H = 2 * ET*std::tan(alpha);
	double W = 2 * ET*std::tan(beta);
	double pix_dh = H / M;
	double pix_hw = W / N;

	

	// 计算A点坐标
	Point A = T - W / 2 * VTQ.normalized() + H / 2 * U.normalized();
	//cout << "A:" << endl << A << endl;

	cv::Mat img(M, N, CV_8UC3);
	assert(img.channels() == 3);

	// 平面ABCD上任意一点的坐标 P(i,j)
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			Point Pij = A - pix_dh / 2 * U.normalized() + pix_hw / 2 * VTQ.normalized() -
				i * pix_dh*U.normalized() + j * pix_hw*VTQ.normalized();
			auto c = GetPixel(images, Pij);
			img.at<cv::Vec3b>(i, j) = c;
		}
	}
	

	return img;
}
