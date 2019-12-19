#include <iostream>
#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <map>

constexpr int ET = 1;
constexpr double PI = 3.1415926;
using namespace std;

#define TX 32
#define TY 32

// 正方体表面索引编号
#define UP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3
#define FRONT 4
#define BACK 5

/****************************************************************************************************
 *
 *  辅助数据结构
 *
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
 *   自定义图像类以适配CUDA
 */
struct GImage
{
	int rows;
	int cols;
	Vector3d *data;
};

/*
 *  获取图像指定位置的像素值
 */
__host__ __device__
Vector3d GPixel(const GImage *pimg, int row, int col)
{
	return pimg->data[col + row * pimg->cols];
}

/*
 *  设置图像指定位置的像素值
 */
__host__ __device__
void SetGPixel(GImage *pimg, int row, int col, Vector3d new_pixel)
{
	pimg->data[col + row * pimg->cols] = new_pixel;
}

/****************************************************************************************************
 *
 *  主流程相关函数
 *
 ***************************************************************************************************/

/*
 *  计算指定坐标点 pt 与正方体哪个面相交，并计算相交的坐标位置，保存于参数 intersect_point 内
 */
__device__
int WhichPlane(Vector3d pt, Vector3d &intersect_point)
{
	// 同方向的单位向量
	auto nor = Normalized(&pt);

	// 判断 pt 是否与上面相交
	if (Index(&nor, 2) > 0)  // 若相交则 z 坐标必定大于 0
	{
		double scale = HALF_SIDE / std::abs(Index(&nor, 2)); // 将单位方向向量 z 坐标缩放到与正方体上面相交的长度，并保存作坊比例
		double x, y;
		x = Index(&nor, 0)*scale;  // 计算缩放后的 x 坐标
		y = Index(&nor, 1)*scale;  // 计算缩放后的 y 坐标


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)  // 若相交，则 x,y 坐标必定在上面的矩形区域内
		{
#ifdef _DEBUG
			//std::cout << "上" << std::endl;
			printf("上\n");
#endif
			SetVector3d(&intersect_point, x, y, HALF_SIDE);
			return UP;
		}


	}

	/*
	 *    其他面的判断方式和上面类似
	 */

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


/*
 *  计算由坐标系原点出发，经过指定坐标 pt 发出的射线与正方体其中一个面相交的位置处的图像像素值
 */
__device__
const Vector3d GetPixel(const GImage images[], Vector3d pt)
{
	// 交点
	Vector3d intersect_point;

	int M = images[UP].rows;
	int N = images[UP].cols;

	// 计算相交的平面索引与交点位置
	int plane_index = WhichPlane(pt, intersect_point);

	int i = 0, j = 0;
	switch (plane_index)
	{
		/*
		 *  由正方体的内部向外看去，z 轴指向上面，与上面相交
		 *  x 轴和 y 轴分别表示上面图像的行列索引，且上面起始坐标为 (-0.5, 0.5, 0.5)
		 *  x的坐标范围为(-0.5,0.5), 对于x任意值 tmp_x, 其距离 x 的原点的距离为 0.5+tmp_x, 如 x_tmp = -0.2, 则distance = 0.5 +（-0.2）= 0.3
		 *  y的坐标范围为(0.5,-0.5), 对于y任意值 tmp_y, 其距离 y 的原点的距离为 0.5-tmp_x, 如 x_tmp = -0.2, 则distance = 0.5 -（-0.2）= 0.7
		 */
	case UP:
		i = (HALF_SIDE + Index(&intersect_point, 0))*M;  // HALF_SIDE + Index(&intersect_point, 0) 为距离原点的比例，
		                                                 // M为资源图像的高度
		                                                 // 二者相乘，则可得该位置对应资源图像上像素的对应位置
		j = (HALF_SIDE - Index(&intersect_point, 1))*N;

		if (i >= M)i = M - 1;                            // 防止索引越界
		if (j >= N)j = N - 1;

		return GPixel(&images[UP], i, j);                // 根据图像映射位置，获取该位置的实际像素值
		
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
void ComputePixelValue(Point *A,Vector3d *U, float pix_dh, float pix_hw, const Vector3d *VTQ, const GImage images[], GImage *result)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i < result->rows && j < result->cols)
	{
		// 像素坐标
		Point Pij = (*A) - U->normalized()*pix_dh / 2.0   + VTQ->normalized()*pix_hw / 2.0   - U->normalized()*i * pix_dh + VTQ->normalized()*j * pix_hw;
		
		// 与平面相交的交点位置像素值
		auto c = GetPixel(images, Pij);

		// 设置像素值
		SetGPixel(result, i, j, c);
	}
	
}



void GraphGenerate(const std::map<int, cv::Mat> &images_cpu, Vector3d U_cpu, Vector3d F_cpu, double alpha, double beta, int M, int N, cv::Mat &result_cpu)
{
	// 计算 T 点坐标
	auto nor_F = Normalized(&F_cpu);
	Point T = VecMultifloat(&nor_F, ET);
	//cout << "T:" << endl << T << endl;
	//Print(T);

	// 垂直于平面ETU的法向量
	auto VTQ = Cross(&F_cpu, &U_cpu);
	//Print(VTQ);

	// 图像宽和高
	float H = 2 * ET*std::tan(alpha * PI / 180);
	float W = 2 * ET*std::tan(beta * PI / 180);
	float pix_dh = H / M;
	float pix_hw = W / N;



	// 计算A点坐标
	Point A = T - VTQ.normalized()*W / 2   + U_cpu.normalized()*H / 2  ;
	//Print(A);
	

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
	
	
	GImage *result_gpu = 0;
	cudaMallocManaged(&result_gpu, sizeof(GImage));
	result_gpu->rows = M;
	result_gpu->cols = N;
	cudaMallocManaged(&result_gpu->data, M*N * sizeof(Vector3d));
	
	const dim3 blockSize(TX, TY);
	const int bx = (M + TX - 1) / TX;
	const int by = (N + TY - 1) / TY;
	const dim3 gridSize = dim3(bx, by);

	// 计时开始
	cudaEvent_t start_gpu = 0, stop_gpu = 0;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
	cudaEventRecord(start_gpu);

	ComputePixelValue << <gridSize, blockSize >> > (A_GPU, U_GPU, pix_dh, pix_hw, VTQ_GPU, images_gpu, result_gpu);
	cudaDeviceSynchronize();

	// 计时结束
	cudaEventRecord(stop_gpu);
	cudaEventSynchronize(stop_gpu);
	float time_matrix_add_gpu = 0;
	cudaEventElapsedTime(&time_matrix_add_gpu, start_gpu, stop_gpu);
	std::cout << "consumed time: " << time_matrix_add_gpu << " ms" << std::endl;

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

	// 释放内存
	cudaFree(A_GPU);
	cudaFree(U_GPU);
	cudaFree(VTQ_GPU);

	for (int i = 0; i < 6; i++)
	{
		cudaFree(images_gpu[i].data);
	}
	cudaFree(images_gpu);
	cudaFree(result_gpu->data);
	cudaFree(result_gpu);
}


int main()
{
	Vector3d U;
	SetVector3d(&U, 0, 0, 1);
	Vector3d F;
	SetVector3d(&F, 1, 1, 0);

	// 读取图片
	cv::Mat image_up = cv::imread("../data/up.jpg");
	cv::Mat image_bottom = cv::imread("../data/bottom.jpg");
	cv::Mat image_front = cv::imread("../data/front.jpg");
	cv::Mat image_back = cv::imread("../data/back.jpg");
	cv::Mat image_left = cv::imread("../data/left.jpg");
	cv::Mat image_right = cv::imread("../data/right.jpg");

	
	std::map<int, cv::Mat> imgs;
	imgs[UP] = image_up;
	imgs[BOTTOM] = image_bottom;
	imgs[LEFT] = image_left;
	imgs[RIGHT] = image_right;
	imgs[FRONT] = image_front;
	imgs[BACK] = image_back;

	cv::Mat result;
	/*
	 *  参数三和参数四角度的单位 为 度(°)，而非弧度
	 */
	GraphGenerate(imgs, U, F, 30 , 40 , 768, 1024, result);
	cudaDeviceSynchronize();

	cv::imshow("result", result);
	cv::waitKey();
	

	
#ifdef _WIN32
	system("pause");
#endif
    return 0;
}



