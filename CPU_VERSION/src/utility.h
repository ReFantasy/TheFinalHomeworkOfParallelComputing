#ifndef __UTILITY_H__
#define __UTILITY_H__
#include <iostream>
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>

enum class PlaneIndex{ UP, BOTTOM, LEFT, RIGHT, FRONT, BACK,NONE };

class Vector3d
{
	friend Vector3d operator*(const Vector3d v, float d);
	friend Vector3d operator*(float d, const Vector3d v);
	friend Vector3d operator-(const Vector3d v1, const Vector3d v2);
	friend Vector3d operator+(const Vector3d v1, const Vector3d v2);
public:
	Vector3d() = default;
	Vector3d(float x, float y, float z) :_x(x), _y(y), _z(z) {}
	Vector3d normalized()const
	{
		float _mul = mul();
		return Vector3d{ _x / _mul, _y / _mul,_z / _mul };
	}
	Vector3d cross(const Vector3d u)
	{
		/*auto i = Vector3d(1, 0, 0);
		auto j = Vector3d(0, 1, 0);
		auto k = Vector3d(0, 0, 1);
		float a1 = _x;
		float b1 = _y;
		float c1 = _z;
		float a2 = u._x;
		float b2 = u._y;
		float c2 = u._z;

		return (b1*c2 - b2 * c1)*i + (a2*c1 - a1 * c2)*j + (a1*b2 - a2 * b1)*k;*/

		float a1 = _x;
		float b1 = _y;
		float c1 = _z;
		float a2 = u._x;
		float b2 = u._y;
		float c2 = u._z;

		return Vector3d(b1*c2 - b2 * c1, a2*c1 - a1 * c2, a1*b2 - a2 * b1);
	}
	


	float operator()(int i)
	{
		if (i == 0)
			return _x;
		else if (i == 1)
			return _y;
		else if (i == 2)
			return _z;
		else
			return 0;
	}
private:
	float mul()const
	{
		return std::pow(_x*_x + _y * _y + _z * _z, 0.5);
	}

private:
	float _x = 0;
	float _y = 0;
	float _z = 0;
};

Vector3d operator*(const Vector3d v, float d);

Vector3d operator*(float d, const Vector3d v);

Vector3d operator-(const Vector3d v1, const Vector3d v2);
Vector3d operator+(const Vector3d v1, const Vector3d v2);

/*
 * �ж����� OP ���������ཻ��ƽ��
 * 
 */

PlaneIndex WhichPlane(Vector3d pt, Vector3d &intersect_point);

// ��ȡ������������������ɫ
const cv::Vec3b GetPixel(const std::map<PlaneIndex, cv::Mat> &images, Vector3d pt);


#endif//__UTILITY_H__