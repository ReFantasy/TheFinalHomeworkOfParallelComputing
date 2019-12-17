#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "utility.h"
#include <map>

using namespace Eigen;
constexpr int ET = 1;
constexpr double PI = 3.1415926;
using namespace std;
using Point = Eigen::Vector3d;


cv::Mat ViewGenerate(Eigen::Vector3d U, Eigen::Vector3d F, double alpha, double beta, int M, int N);
std::map<PlaneIndex, cv::Mat> images;

cv::Vec3b f()
{
	return images.find(PlaneIndex::UP)->second.at<cv::Vec3b>(10, 10);
}
int main()
{
	
	Eigen::Vector3d U{ 0,0,1 };
	Eigen::Vector3d F{ 0,1,0 };
	//ViewGenerate(U, F, 45 * PI / 180, 45 * PI / 180, 100, 100);

	/*Eigen::Vector3d intersect_point;
	Eigen::Vector3d P{ 0,0.6,0.49 };
	WhichPlane(P, intersect_point);
	cout << intersect_point.transpose() << endl;*/
	
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

	
	//cout << v << endl;

	auto img = ViewGenerate(U, F, 45 * PI / 180.0, 30 * PI / 180.0, 800, 800);
	img.resize(600);
	cv::resize(img, img, cv::Size(600, 600));
	cv::imshow("result", img);
	cv::waitKey();
	


	
#ifdef _WIN32
	system("pause");
#endif
    return 0;
}


cv::Mat ViewGenerate(Eigen::Vector3d U, Eigen::Vector3d F, double alpha, double beta, int M, int N)
{
	// 计算 T 点坐标
	
	
	Point T = F / F.norm()*ET;
	cout << "T:" << endl << T << endl;

	// 垂直于平面ETU的法向量
	auto VTQ = F.cross(U);

	// 图像宽和高
	double H = 2 * ET*std::tan(alpha);
	double W = 2 * ET*std::tan(beta);
	double pix_dh = H / M;
	double pix_hw = W / N;

	// 计算S坐标
	Point S = T + H / 2 * U.normalized();
	cout << "S:" << endl << S << endl;

	// 计算Q点坐标
	Point Q = T + W / 2 * VTQ.normalized();
	cout << "Q:" << endl << Q << endl;

	// 计算A点坐标

	Point A = T - W / 2 * VTQ.normalized() + H / 2 * U.normalized();
	cout << "A:" << endl << A << endl;

	cv::Mat img(M, N, CV_8UC3);
	assert(img.channels() == 3);

	// 平面ABCD上任意一点的坐标 P(i,j)
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			Point Pij = A - pix_dh / 2 * U.normalized() + pix_hw / 2 * VTQ.normalized() -
				i * pix_dh*U.normalized() + j * pix_hw*VTQ.normalized();
			auto c = GetPixel(images, Pij, M, N);
			img.at<cv::Vec3b>(i, j) = c;
		}
	}
	

	return img;
}
