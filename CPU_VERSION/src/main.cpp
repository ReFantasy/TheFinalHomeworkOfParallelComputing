/*
 *
 *  main program
 *
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include <map>
#include <chrono>

constexpr int ET = 1;
constexpr double PI = 3.141592653;

using namespace std;
using namespace std::chrono;
using Point = Vector3d;

/** \brief Generate graph to specific view
 *  \param[in] images Contains the resource image of the cube's six faces
 *  \param[in] U Specifies the upward direction of the field of view, perpendicularing to the F
 *  \param[in] F Specifies the forward direction of the field of view, perpendicularing to the U
 *  \param[in] alpha Half Angle of vertical field of view, alpha<90
 *  \param[in] beta Half Angle of horizontal field of view, beta<90
 *  \param[in] M The height of the return graph, the recommendation is proportional to the field of view
 *  \param[in] N The width of the return graph, the recommendation is proportional to the field of view
 *  \return the graph to the view
 */
cv::Mat GraphGenerate(const std::map<PlaneIndex, cv::Mat> &images, Vector3d U, Vector3d F, double alpha, double beta, int M, int N);

int main()
{
	// Read the resource file
	cv::Mat image_up = cv::imread("../data/up.jpg");
	cv::Mat image_bottom = cv::imread("../data/bottom.jpg");
	cv::Mat image_front = cv::imread("../data/front.jpg");
	cv::Mat image_back = cv::imread("../data/back.jpg");
	cv::Mat image_left = cv::imread("../data/left.jpg");
	cv::Mat image_right = cv::imread("../data/right.jpg");

	std::map<PlaneIndex, cv::Mat> images;
	images[PlaneIndex::UP] = image_up;
	images[PlaneIndex::BOTTOM] = image_bottom;
	images[PlaneIndex::FRONT] = image_front;
	images[PlaneIndex::BACK] = image_back;
	images[PlaneIndex::LEFT] = image_left;
	images[PlaneIndex::RIGHT] = image_right;

	// Set view direction
	Vector3d U(-1, -1, 2);
	Vector3d F(1, 1, 1);

	auto start_time = high_resolution_clock::now();

	/*
	 * Compute the graph of the specific view.
	 * Parameters three and four are in degrees, not radians.
	 */
	auto img = GraphGenerate(images, U, F, 45 , 45 , 1024, 1024);


	auto end_time = high_resolution_clock::now();

	std::cout << "consumed time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms" << std::endl;

	// show results
	cv::imshow("result", img);
	cv::waitKey();
	


#ifdef _WIN32
	system("pause");
#endif
    return 0;
}


cv::Mat GraphGenerate(const std::map<PlaneIndex, cv::Mat> &images, Vector3d U, Vector3d F, double alpha, double beta, int M, int N)
{
	Point T = F.normalized()*ET;
	//cout << "T:" << endl << T << endl;

	// the normal vector to the plane which is consist of vector F and vector U
	auto VTQ = F.cross(U);

	// the width and height of the view
	double H = 2 * ET*std::tan(alpha * PI / 180.0);
	double W = 2 * ET*std::tan(beta * PI / 180.0);
	double pix_dh = H / M;
	double pix_hw = W / N;

	
	Point A = T - W / 2 * VTQ.normalized() + H / 2 * U.normalized();
	//cout << "A:" << endl << A << endl;

	cv::Mat result_graph(M, N, CV_8UC3);
	assert(result_graph.channels() == 3);  // must be RGB image 

	/*
	 *  the coordinates of any point to the plane ABCD.
	 *  we can get any pixel's size according to M and N,
	 *  and then we go through all the pixels from point A.
	 *  we can get pixel's coordinates according to : A + pixel_index * pixel_size
	 */
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			Point p_i_j = A - pix_dh / 2 * U.normalized() + pix_hw / 2 * VTQ.normalized() -
				i * pix_dh*U.normalized() + j * pix_hw*VTQ.normalized();

			auto pixel_value = GetPixel(images, p_i_j);
			result_graph.at<cv::Vec3b>(i, j) = pixel_value;
		}
	}
	

	return result_graph;
}