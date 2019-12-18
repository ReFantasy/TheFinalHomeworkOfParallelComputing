#include "utility.h"

// 正方体边长的一半
const double HALF_SIDE = 0.5;

Vector3d operator*(const Vector3d v, float d)
{
	return Vector3d{ v._x*d, v._y*d, v._z*d };
}

Vector3d operator*(float d, const Vector3d v)
{
	return v * d;
}

Vector3d operator-(const Vector3d v1, const Vector3d v2)
{
	return Vector3d{ v1._x - v2._x, v1._y - v2._y, v1._z - v2._z };
}

Vector3d operator+(const Vector3d v1, const Vector3d v2)
{
	return Vector3d{ v1._x + v2._x, v1._y + v2._y, v1._z + v2._z };
}

PlaneIndex WhichPlane(Vector3d pt, Vector3d &intersect_point)
{
	auto nor = pt.normalized();

	// "上"
	if (nor(2) > 0)
	{
		double scale = HALF_SIDE / std::abs(nor(2));
		double x, y;
		x = nor(0)*scale;
		y = nor(1)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "上" << std::endl;
#endif
			intersect_point = Vector3d( x,y,HALF_SIDE );
			return PlaneIndex::UP;
		}

		
	}

	// "下"
	if (nor(2) < 0)
	{
		double scale = HALF_SIDE / std::abs(nor(2));
		double x, y;
		x = nor(0)*scale;
		y = nor(1)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "下" << std::endl;
#endif
			intersect_point = Vector3d( x,y,-HALF_SIDE );
			return PlaneIndex::BOTTOM;
		}

		
	}

	// "前"
	if (nor(0) > 0)
	{
		double scale = HALF_SIDE / std::abs(nor(0));
		double y, z;
		y = nor(1)*scale;
		z = nor(2)*scale;


		if (std::abs(y) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "前" << std::endl;
#endif
			intersect_point = Vector3d( HALF_SIDE,y,z );
			return PlaneIndex::FRONT;
		}

		
	}

	// "后"
	if (nor(0) < 0)
	{
		double scale = HALF_SIDE / std::abs(nor(0));
		double y, z;
		y = nor(1)*scale;
		z = nor(2)*scale;


		if (std::abs(y) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "后" << std::endl;
#endif
			intersect_point = Vector3d( -HALF_SIDE,y,z );
			return PlaneIndex::BACK;
		}

		
	}

	// "左"
	if (nor(1) > 0)
	{
		double scale = HALF_SIDE / std::abs(nor(1));
		double x, z;
		x = nor(0)*scale;
		z = nor(2)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "左" << std::endl;
#endif
			intersect_point = Vector3d( x,-HALF_SIDE,z );
			return PlaneIndex::LEFT;
		}

		
	}

	// "右"
	if (nor(1) < 0)
	{
		double scale = HALF_SIDE / std::abs(nor(1));
		double x, z;
		x = nor(0)*scale;
		z = nor(2)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
#ifdef _DEBUG
			//std::cout << "右" << std::endl;
#endif
			intersect_point = Vector3d( x,HALF_SIDE,z );
			return PlaneIndex::RIGHT;
		}

		
	}


	return PlaneIndex::NONE;
}

const cv::Vec3b GetPixel(const std::map<PlaneIndex, cv::Mat> &images, Vector3d pt)
{
	// 交点
	Vector3d intersect_point;

	int M = images.find(PlaneIndex::UP)->second.rows;
	int N = images.find(PlaneIndex::UP)->second.cols;

	// 相交的平面
	PlaneIndex plane = WhichPlane(pt, intersect_point);
	int i = 0, j = 0;

	switch (plane)
	{
	case PlaneIndex::UP:
		i = (HALF_SIDE + intersect_point(0))*M;
		j = (HALF_SIDE - intersect_point(1))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return images.find(PlaneIndex::UP)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::BOTTOM:
		i = (HALF_SIDE - intersect_point(0))*M;
		j = (HALF_SIDE - intersect_point(1))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return images.find(PlaneIndex::BOTTOM)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::LEFT: 
		i = (HALF_SIDE - intersect_point(2))*M;
		j = (HALF_SIDE + intersect_point(0))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return images.find(PlaneIndex::LEFT)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::RIGHT:
		i = (HALF_SIDE - intersect_point(2))*M;
		j = (HALF_SIDE - intersect_point(0))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return images.find(PlaneIndex::RIGHT)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::FRONT:
		j = (HALF_SIDE - intersect_point(1))*M;
		i = (HALF_SIDE - intersect_point(2))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return images.find(PlaneIndex::FRONT)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::BACK:
		i = (HALF_SIDE - intersect_point(2))*M;
		j = (HALF_SIDE + intersect_point(1))*N;
		if (i >= M)i = M - 1;
		if (j >= N)j = N - 1;
		return images.find(PlaneIndex::BACK)->second.at<cv::Vec3b>(i, j);
		break;


	default:
		break;
	}

	return cv::Vec3b{};
}
