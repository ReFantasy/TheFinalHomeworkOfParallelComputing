#include <iostream>
#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "utility.h"
#include <map>


constexpr int ET = 1;
constexpr double PI = 3.1415926;
using namespace std;

#define TX 32
#define TY 32

#define UP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3
#define FRONT 4
#define BACK 5

/****************************************************************************************************
 *  定义相关数据结构
 ***************************************************************************************************/

/*
 *  三维向量
 */
struct Vector3d
{
	float _x = 0;
	float _y = 0;
	float _z = 0;
	Vector3d() = default;
	Vector3d(float x, float y, float z) :_x(x), _y(y), _z(z) {}
	__host__ __device__
	Vector3d operator*(float d)
	{
		Vector3d tmp;
		tmp._x = _x * d;
		tmp._y = _y * d;
		tmp._z = _z * d;
		return tmp;
	}

	__host__ __device__
	Vector3d operator/(float d)
	{
		Vector3d tmp;
		tmp._x = _x / d;
		tmp._y = _y / d;
		tmp._z = _z / d;
		return tmp;
	}

	__host__ __device__
	Vector3d operator-(const  Vector3d &d)
	{
		Vector3d tmp;
		tmp._x = _x - d._x;
		tmp._y = _y - d._y;
		tmp._z = _z - d._z;
		return tmp;
	}

	__host__ __device__
	Vector3d operator+(const  Vector3d &d)
	{
		Vector3d tmp;
		tmp._x = _x + d._x;
		tmp._y = _y + d._y;
		tmp._z = _z + d._z;
		return tmp;
	}

	__host__ __device__
	float mul()const
	{

		return sqrt(_x*_x + _y * _y + _z * _z);

	}

	__host__ __device__
	Vector3d normalized()const
	{

		float _mul = mul();
		return Vector3d{ _x / _mul, _y / _mul,_z / _mul };

	}

};
using Point = Vector3d;

// 正方体边长的一半
//const double HALF_SIDE = 0.5;
#define HALF_SIDE (0.5)
__host__ __device__
void Print(Vector3d v)
{
	printf("(%2f,%2f,%2f)\n", v._x, v._y, v._z);
}

__host__ __device__
float Index(const Vector3d *pv, int index)
{
	if (index == 0)
		return pv->_x;
	else if (index == 1)
		return pv->_y;
	else if (index == 2)
		return pv->_z;
	else
		return 0;
}

__host__ __device__
void SetVector3d(Vector3d *pv, float x, float y, float z)
{
	pv->_x = x;
	pv->_y = y;
	pv->_z = z;
}


__host__ __device__
float Mul(const Vector3d *pv)
{
	return sqrt(pv->_x*pv->_x + pv->_y * pv->_y + pv->_z * pv->_z);
}

__host__ __device__
Vector3d Normalized(const Vector3d *pv)
{
	float _mul = Mul(pv);
	Vector3d res;
	res._x = pv->_x / _mul;
	res._y = pv->_y / _mul;
	res._z = pv->_z / _mul;
	return res;
}

__host__ __device__
Vector3d Cross(const Vector3d *u, const Vector3d *v)
{
	float a1 = u->_x;
	float b1 = u->_y;
	float c1 = u->_z;
	float a2 = v->_x;
	float b2 = v->_y;
	float c2 = v->_z;

	Vector3d res;
	res._x = b1 * c2 - b2 * c1;
	res._y = a2 * c1 - a1 * c2;
	res._z = a1 * b2 - a2 * b1;
	return res;
}

__host__ __device__
Vector3d VecMultifloat(const Vector3d *pv, float d)
{
	Vector3d res;
	res._x = pv->_x*d;
	res._y = pv->_y*d;
	res._z = pv->_z*d;
	return res;
}

__host__ __device__
Vector3d floatMultiVec(float d, const Vector3d *pv)
{
	return VecMultifloat(pv, d);
}



__host__ __device__
Vector3d Sub(const Vector3d *pv1, const Vector3d *pv2)
{
	Vector3d res;
	res._x = pv1->_x - pv2->_x;
	res._y = pv1->_y - pv2->_y;
	res._z = pv1->_z - pv2->_z;
	return res;
}

__host__ __device__
Vector3d Add(const Vector3d *pv1, const Vector3d *pv2)
{
	Vector3d res;
	res._x = pv1->_x + pv2->_x;
	res._y = pv1->_y + pv2->_y;
	res._z = pv1->_z + pv2->_z;
	return res;
}


/*
 *   自定义图像类
 */
struct GImage
{
	int rows;
	int cols;
	Vector3d *data;
};

__host__ __device__
Vector3d GPixel(const GImage *pimg, int row, int col)
{
	return pimg->data[col + row * pimg->cols];
}

__host__ __device__
void SetGPixel(GImage *pimg, int row, int col, Vector3d new_pixel)
{
	pimg->data[col + row * pimg->cols] = new_pixel;
}

/****************************************************************************************************
 *  process
 ***************************************************************************************************/
__device__
int WhichPlane(Vector3d pt, Vector3d &intersect_point)
{
	auto nor = Normalized(&pt);

	// "上"
	if (Index(&nor, 2) > 0)
	{
		double scale = HALF_SIDE / std::abs(Index(&nor, 2));
		double x, y;
		x = Index(&nor, 0)*scale;
		y = Index(&nor, 1)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "上" << std::endl;
			printf("上\n");
#endif
			SetVector3d(&intersect_point, x, y, HALF_SIDE);
			return UP;
		}


	}

	// "下"
	if (Index(&nor, 2) < 0)
	{
		double scale = HALF_SIDE / std::abs(Index(&nor, 2));
		double x, y;
		x = Index(&nor, 0)*scale;
		y = Index(&nor, 1)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "下" << std::endl;
			printf("下\n");
#endif
			SetVector3d(&intersect_point, x, y, -HALF_SIDE);
			return BOTTOM;
		}


	}

	// "前"
	if (Index(&nor, 0) > 0)
	{
		double scale = HALF_SIDE / std::abs(Index(&nor, 0));
		double y, z;
		y = Index(&nor, 1)*scale;
		z = Index(&nor, 2)*scale;


		if (std::abs(y) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "前" << std::endl;
			printf("前\n");
#endif
			SetVector3d(&intersect_point, HALF_SIDE, y, z);
			return FRONT;
		}


	}

	// "后"
	if (Index(&nor, 0) < 0)
	{
		double scale = HALF_SIDE / std::abs(Index(&nor, 0));
		double y, z;
		y = Index(&nor, 1)*scale;
		z = Index(&nor, 2)*scale;


		if (std::abs(y) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "后" << std::endl;
			printf("后\n");
#endif
			SetVector3d(&intersect_point, -HALF_SIDE, y, z);
			return BACK;
		}


	}

	// "左"
	if (Index(&nor, 1) > 0)
	{
		double scale = HALF_SIDE / std::abs(Index(&nor, 1));
		double x, z;
		x = Index(&nor, 0)*scale;
		z = Index(&nor, 2)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "左" << std::endl;
			printf("左\n");
#endif
			SetVector3d(&intersect_point, x, -HALF_SIDE, z);
			return LEFT;
		}


	}

	// "右"
	if (Index(&nor, 1) < 0)
	{
		double scale = HALF_SIDE / std::abs(Index(&nor, 1));
		double x, z;
		x = Index(&nor, 0)*scale;
		z = Index(&nor, 2)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "右" << std::endl;
			printf("右\n");
#endif
			SetVector3d(&intersect_point, x, HALF_SIDE, z);
			return RIGHT;
		}


	}


	return -1;
}


__device__
const Vector3d GetPixel(const GImage images[], Vector3d pt)
{
	// 交点
	Vector3d intersect_point;

	int M = images[UP].rows;
	int N = images[UP].cols;

	// 相交的平面
	int plane = WhichPlane(pt, intersect_point);
	int i = 0, j = 0;

	switch (plane)
	{
	case UP:
		i = (HALF_SIDE + Index(&intersect_point, 0))*M;
		j = (HALF_SIDE - Index(&intersect_point, 1))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return GPixel(&images[UP], i, j);
		
		break;

	case BOTTOM:
		i = (HALF_SIDE - Index(&intersect_point, 0))*M;
		j = (HALF_SIDE - Index(&intersect_point, 1))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return GPixel(&images[BOTTOM], i, j);
		break;

	case LEFT: 
		i = (HALF_SIDE - Index(&intersect_point, 2))*M;
		j = (HALF_SIDE + Index(&intersect_point, 0))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return GPixel(&images[LEFT], i, j);
		break;

	case RIGHT:
		i = (HALF_SIDE - Index(&intersect_point, 2))*M;
		j = (HALF_SIDE - Index(&intersect_point, 0))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return GPixel(&images[RIGHT], i, j);
		break;

	case FRONT:
		j = (HALF_SIDE - Index(&intersect_point, 1))*M;
		i = (HALF_SIDE - Index(&intersect_point, 2))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return GPixel(&images[FRONT], i, j);
		break;

	case BACK:
		i = (HALF_SIDE - Index(&intersect_point, 2))*M;
		j = (HALF_SIDE + Index(&intersect_point, 1))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return GPixel(&images[BACK], i, j);
		break;


	default:
		break;
	}

	return Vector3d{};
}

//cv::Mat ViewGenerate(Vector3d U, Vector3d F, double alpha, double beta, int M, int N);

__global__
void ComputePixelValue(Point *A, Vector3d *U, float pix_dh, float pix_hw, Vector3d *VTQ, GImage images[], GImage *result)
{
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i < result->rows && j < result->cols)
	{
		//Point Pij = Add(&Sub(&Add(&Sub(A, &VecMultifloat(&Normalized(U), pix_dh / 2)), &VecMultifloat(&Normalized(VTQ), pix_hw / 2)), &VecMultifloat(&Normalized(U), i * pix_dh)), &VecMultifloat(&Normalized(VTQ), j * pix_hw));
		
		Point Pij = (*A) - U->normalized()*pix_dh / 2.0   + VTQ->normalized()*pix_hw / 2.0   - U->normalized()*i * pix_dh + VTQ->normalized()*j * pix_hw;
		//printf("row:%d col:%d ", i, j);
		//Print(Pij);
		auto c = GetPixel(images, Pij);
		SetGPixel(result, i, j, c);
		//printf("block:%d  thread:%d\n", i, j);
	}
	
}



void ViewGenerate(const std::map<int, cv::Mat> &images_cpu, Vector3d U_cpu, Vector3d F_cpu, double alpha, double beta, int M, int N, cv::Mat &result_cpu)
{
	// 计算 T 点坐标
	auto nor_F = Normalized(&F_cpu);
	Point T = VecMultifloat(&nor_F, ET);
	//cout << "T:" << endl << T << endl;
	Print(T);

	// 垂直于平面ETU的法向量
	auto VTQ = Cross(&F_cpu, &U_cpu);
	Print(VTQ);

	// 图像宽和高
	float H = 2 * ET*std::tan(alpha);
	float W = 2 * ET*std::tan(beta);
	float pix_dh = H / M;
	float pix_hw = W / N;



	// 计算A点坐标
	//Point A = T - Normalized(&VTQ)*W / 2.0 + Normalized(&U)* H / 2 ;
	
	//
	/*auto nor_vtq = Normalized(&VTQ);
	auto nor_u = Normalized(&U_cpu);
	auto v1 = VecMultifloat(&nor_vtq, W / 2.0);
	auto v2 = VecMultifloat(&nor_u, H / 2.0);
	Point A = Sub(&Sub(&T, &v1), &v2);*/
	Point A = T - VTQ.normalized()*W / 2   + U_cpu.normalized()*H / 2  ;

	Print(A);
	//cout << "A:" << endl << A << endl;

	// 申请GPU端数据
	Point *A_GPU = 0;
	cudaMallocManaged(&A_GPU, sizeof(Point));
	SetVector3d(A_GPU, A._x, A._y, A._z);

	Point *U_GPU = 0;
	cudaMallocManaged(&U_GPU, sizeof(Point));
	SetVector3d(U_GPU, U_cpu._x, U_cpu._y, U_cpu._z);

	Point *VTQ_GPU = 0;
	cudaMallocManaged(&VTQ_GPU, sizeof(Point));
	SetVector3d(VTQ_GPU, VTQ._x, VTQ._y, VTQ._z);

	GImage *images_gpu = 0;
	cudaMallocManaged(&images_gpu, sizeof(GImage)*6);
	int rows = images_cpu.find(0)->second.rows;
	int cols = images_cpu.find(0)->second.cols;
	for (int i = 0; i < 6; i++)
	{
		images_gpu[UP + i].rows = rows;
		images_gpu[UP + i].cols = cols;
		cudaMallocManaged(&images_gpu[UP + i].data, rows*cols*sizeof(Vector3d));
		for (int m = 0; m < rows; m++)
		{
			for (int n = 0; n < cols; n++)
			{
				Vector3d pix_value;
				SetVector3d(&pix_value, images_cpu.find(UP + i)->second.at<cv::Vec3b>(m,n)[0],
					images_cpu.find(UP + i)->second.at<cv::Vec3b>(m, n)[1],
					images_cpu.find(UP + i)->second.at<cv::Vec3b>(m, n)[2]);//images_cpu.find(UP + i)->second.
				SetGPixel(&images_gpu[UP + i], m, n, pix_value);
			}
		}

	}
	
	

    printf("ViewGenerate...\n");
	GImage *result_gpu = 0;
	cudaMallocManaged(&result_gpu, sizeof(GImage));
	result_gpu->rows = M;
	result_gpu->cols = N;
	cudaMallocManaged(&result_gpu->data, M*N * sizeof(Vector3d));
	
	const dim3 blockSize(TX, TY);
	const int bx = (M + TX - 1) / TX;
	const int by = (N + TY - 1) / TY;
	const dim3 gridSize = dim3(bx, by);
	ComputePixelValue << <gridSize, blockSize >> > (A_GPU, U_GPU, pix_dh, pix_hw, VTQ_GPU, images_gpu, result_gpu);
	cudaDeviceSynchronize();

	result_cpu = cv::Mat(M, N, CV_8UC3);
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			Vector3d value = GPixel(result_gpu, i, j);
			cv::Vec3b v;
			v[0] = value._x;
			v[1] = value._y;
			v[2] = value._z;
			result_cpu.at<cv::Vec3b>(i, j) = v;
		}
	}
}


int main()
{
	Vector3d U;
	SetVector3d(&U, 0, 0, 1);
	Vector3d F;
	SetVector3d(&F, 1, 1, 0);

	// 读取图片
	cv::Mat image_up = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\up.jpg");
	cv::Mat image_bottom = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\bottom.jpg");
	cv::Mat image_front = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\front.jpg");
	cv::Mat image_back = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\back.jpg");
	cv::Mat image_left = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\left.jpg");
	cv::Mat image_right = cv::imread("F:\\homework\\parallel computing\\FinallyHomework\\Code\\data\\right.jpg");
	std::map<int, cv::Mat> imgs;
	imgs[UP] = image_up;
	imgs[BOTTOM] = image_bottom;
	imgs[LEFT] = image_left;
	imgs[RIGHT] = image_right;
	imgs[FRONT] = image_front;
	imgs[BACK] = image_back;

	cv::Mat result;
	ViewGenerate(imgs, U, F, 30 * PI / 180, 40 * PI / 180, 768, 1024, result);
	cudaDeviceSynchronize();

	cv::imshow("ok", result);
	cv::waitKey();
	

	
#ifdef _WIN32
	system("pause");
#endif
    return 0;
}



