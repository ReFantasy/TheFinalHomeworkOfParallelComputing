/*
 *
 *  This file contains some generic functions and classes
 *
 */
#ifndef __UTILITY_H__
#define __UTILITY_H__
#include <iostream>
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>

enum class PlaneIndex{ UP, BOTTOM, LEFT, RIGHT, FRONT, BACK,NONE };

/*
 *
 *  A class to represents a three-dimensional vector.
 *  This class defines some basic operations, 
 *  such as vector addition, subtraction, multiplication and division, and modular operations.
 *  Of course, you can use open source libraries such as EIGEN instead. But then you can not
 *  modify this program to adapt to CUDA.
 */
class Vector3d
{
	friend Vector3d operator*(const Vector3d v, float d);
	friend Vector3d operator*(float d, const Vector3d v);
	friend Vector3d operator-(const Vector3d v1, const Vector3d v2);
	friend Vector3d operator+(const Vector3d v1, const Vector3d v2);

public:
	Vector3d() = default;
	Vector3d(float x, float y, float z) :_x(x), _y(y), _z(z) {}

	// Get the unit vector
	Vector3d normalized()const
	{
		float _mul = mul();
		return Vector3d{ _x / _mul, _y / _mul,_z / _mul };
	}

	// Vector cross-product
	Vector3d cross(const Vector3d u)
	{
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

/** \brief Determine which surface of the cube the line between point pt and origin go through
 *  \param[in] pt Pixel point coordinates
 *  \param[out] intersect_point 
 *  \return plane index
 */
PlaneIndex WhichPlane(Vector3d pt, Vector3d &intersect_point);

/** \brief Calculate the pixel value corresponding to point pt
 *  \param[in] images Cube surface image
 *  \param[in] pt The coordinates of point
 *  \return Pixel values
 */
const cv::Vec3b GetPixel(const std::map<PlaneIndex, cv::Mat> &images, Vector3d pt);


#endif//__UTILITY_H__