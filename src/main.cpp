#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "utility.h"

using namespace Eigen;
constexpr int ET = 1;
constexpr double PI = 3.1415926;
using namespace std;
using Point = Eigen::Vector3d;


cv::Mat ViewGenerate(Eigen::Vector3d U, Eigen::Vector3d F, double alpha, double beta, int M, int N);
int main()
{
	
	Eigen::Vector3d U{ 0,0,1 };
	Eigen::Vector3d F{ 0,1,0 };
	//ViewGenerate(U, F, 45 * PI / 180, 45 * PI / 180, 100, 100);

	/*Eigen::Vector3d intersect_point;
	Eigen::Vector3d P{ 0,0.6,0.49 };
	WhichPlane(P, intersect_point);
	cout << intersect_point.transpose() << endl;*/
	

	cv::Mat image_up = cv::imread("./data/up.jpg");
	cv::imshow("up", image_up);
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
	//cout << "T:" << T.transpose() << endl;

	// 垂直于平面ETU的法向量
	auto VTQ = F.cross(U);

	// 图像宽和高
	double H = 2 * ET*std::tan(alpha);
	double W = 2 * ET*std::tan(beta);
	double pix_dh = H / M;
	double pix_hw = W / N;

	// 计算S坐标
	Point S = T + H / 2 * U.normalized();
	//cout << "S:" << S.transpose() << endl;

	// 计算Q点坐标
	Point Q = T + W / 2 * VTQ.normalized();
	//cout << "Q:" << Q.transpose() << endl;

	// 计算A点坐标

	Point A = T - W / 2 * VTQ.normalized() + H / 2 * U.normalized();
	//cout << "A:" << A.transpose() << endl;

	// 平面ABCD上任意一点的坐标 P(i,j)
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			Point Pij = A - pix_dh / 2 * U.normalized() + pix_hw / 2 * VTQ.normalized() -
				i * pix_dh*U.normalized() + j * pix_hw*VTQ.normalized();

			if (i == 0 && j == 0)
			{
				//cout << "Pij:" << Pij.transpose() << endl;
			}
		}
	}
	

	return cv::Mat{};
}
