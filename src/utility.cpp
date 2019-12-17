#include "utility.h"

// ������߳���һ��
const double HALF_SIDE = 0.5;

PlaneIDA WhichPlane(Eigen::Vector3d pt, Eigen::Vector3d &intersect_point)
{
	auto nor = pt.normalized();

	// "��"
	if (nor(2) > 0)
	{
		double scale = HALF_SIDE / nor(2);
		double x, y;
		x = nor(0)*scale;
		y = nor(1)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)
		{
			std::cout << "��" << std::endl;
		}
		
		intersect_point = Eigen::Vector3d{ x,y,HALF_SIDE };
	}

	// "��"
	if (nor(2) < 0)
	{
		double scale = HALF_SIDE / nor(2);
		double x, y;
		x = nor(0)*scale;
		y = nor(1)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(y) <= HALF_SIDE)
		{
			std::cout << "��" << std::endl;
		}
		
		intersect_point = Eigen::Vector3d{ x,y,-HALF_SIDE };
	}

	// "ǰ"
	if (nor(0) > 0)
	{
		double scale = HALF_SIDE / nor(0);
		double y, z;
		y = nor(1)*scale;
		z = nor(2)*scale;


		if (std::abs(y) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
			std::cout << "ǰ" << std::endl;
		}
		
		intersect_point = Eigen::Vector3d{ HALF_SIDE,y,z };
	}

	// "��"
	if (nor(0) < 0)
	{
		double scale = HALF_SIDE / nor(0);
		double y, z;
		y = nor(1)*scale;
		z = nor(2)*scale;


		if (std::abs(y) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
			std::cout << "��" << std::endl;
		}

		intersect_point = Eigen::Vector3d{ -HALF_SIDE,y,z };
	}

	// "��"
	if (nor(1) < 0)
	{
		double scale = HALF_SIDE / nor(1);
		double x, z;
		x = nor(0)*scale;
		z = nor(2)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
			std::cout << "��" << std::endl;
		}

		intersect_point = Eigen::Vector3d{ x,-HALF_SIDE,z };
	}

	// "��"
	if (nor(1) > 0)
	{
		double scale = HALF_SIDE / nor(1);
		double x, z;
		x = nor(0)*scale;
		z = nor(2)*scale;


		if (std::abs(x) <= HALF_SIDE && std::abs(z) <= HALF_SIDE)
		{
			std::cout << "��" << std::endl;
		}

		intersect_point = Eigen::Vector3d{ x,HALF_SIDE,z };
	}
	

	return PlaneIDA::UP;
}