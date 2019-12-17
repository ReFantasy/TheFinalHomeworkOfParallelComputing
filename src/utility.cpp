#include "utility.h"

// 正方体边长的一半
const double HALF_SIDE = 0.5;

PlaneIDA WhichPlane(Eigen::Vector3d pt, Eigen::Vector3d &intersect_point)
{
	auto nor = pt.normalized();

	// "上"
	if (nor(2) > 0)
	{
		double scale = HALF_SIDE / nor(2);
		double x, y;
		x = nor(0)*scale;
		y = nor(1)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)
		{
			std::cout << "上" << std::endl;
		}
		
		intersect_point = Eigen::Vector3d{ x,y,HALF_SIDE };
	}

	// "下"
	if (nor(2) < 0)
	{
		double scale = HALF_SIDE / nor(2);
		double x, y;
		x = nor(0)*scale;
		y = nor(1)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)
		{
			std::cout << "下" << std::endl;
		}
		
		intersect_point = Eigen::Vector3d{ x,y,-HALF_SIDE };
	}

	// "前"
	if (nor(0) > 0)
	{
		double scale = HALF_SIDE / nor(0);
		double y, z;
		y = nor(1)*scale;
		z = nor(2)*scale;


		if (std::abs(y) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
			std::cout << "前" << std::endl;
		}
		
		intersect_point = Eigen::Vector3d{ HALF_SIDE,y,z };
	}

	// "后"
	if (nor(0) < 0)
	{
		double scale = HALF_SIDE / nor(0);
		double y, z;
		y = nor(1)*scale;
		z = nor(2)*scale;


		if (std::abs(y) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
			std::cout << "后" << std::endl;
		}

		intersect_point = Eigen::Vector3d{ -HALF_SIDE,y,z };
	}

	// "左"
	if (nor(1) < 0)
	{
		double scale = HALF_SIDE / nor(1);
		double x, z;
		x = nor(0)*scale;
		z = nor(2)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
			std::cout << "左" << std::endl;
		}

		intersect_point = Eigen::Vector3d{ x,-HALF_SIDE,z };
	}

	// "右"
	if (nor(1) > 0)
	{
		double scale = HALF_SIDE / nor(1);
		double x, z;
		x = nor(0)*scale;
		z = nor(2)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
			std::cout << "右" << std::endl;
		}

		intersect_point = Eigen::Vector3d{ x,HALF_SIDE,z };
	}
	

	return PlaneIDA::UP;
}