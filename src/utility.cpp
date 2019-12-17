#include "utility.h"

// ������߳���һ��
const double HALF_SIDE = 0.5;

PlaneIndex WhichPlane(Eigen::Vector3d pt, Eigen::Vector3d &intersect_point)
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
#ifdef _DEBUG
			std::cout << "��" << std::endl;
#endif
			intersect_point = Eigen::Vector3d{ x,y,HALF_SIDE };
			return PlaneIndex::UP;
		}

		
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
#ifdef _DEBUG
			std::cout << "��" << std::endl;
#endif
			intersect_point = Eigen::Vector3d{ x,y,-HALF_SIDE };
			return PlaneIndex::BOTTOM;
		}

		
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
#ifdef _DEBUG
			std::cout << "ǰ" << std::endl;
#endif
			intersect_point = Eigen::Vector3d{ HALF_SIDE,y,z };
			return PlaneIndex::FRONT;
		}

		
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
#ifdef _DEBUG
			std::cout << "��" << std::endl;
#endif
			intersect_point = Eigen::Vector3d{ -HALF_SIDE,y,z };
			return PlaneIndex::BACK;
		}

		
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
#ifdef _DEBUG
			std::cout << "��" << std::endl;
#endif
			intersect_point = Eigen::Vector3d{ x,-HALF_SIDE,z };
			return PlaneIndex::LEFT;
		}

		
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
#ifdef _DEBUG
			std::cout << "��" << std::endl;
#endif
			intersect_point = Eigen::Vector3d{ x,HALF_SIDE,z };
			return PlaneIndex::RIGHT;
		}

		
	}


	return PlaneIndex::NONE;
}

const cv::Vec3b GetPixel(const std::map<PlaneIndex, cv::Mat> &images, Eigen::Vector3d pt, int M, int N)
{
	// ����
	Eigen::Vector3d intersect_point;

	// �ཻ��ƽ��
	PlaneIndex plane = WhichPlane(pt, intersect_point);
	int i = 0, j = 0;

	switch (plane)
	{
	case PlaneIndex::UP:
		i = (intersect_point(0) + HALF_SIDE)*M;
		j = (intersect_point(1) + HALF_SIDE)*N;
		return images.find(PlaneIndex::UP)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::BOTTOM:
		i = (HALF_SIDE - intersect_point(0))*M;
		j = (HALF_SIDE + intersect_point(1))*N;
		return images.find(PlaneIndex::BOTTOM)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::LEFT:
		i = (intersect_point(0) + HALF_SIDE)*M;
		j = (HALF_SIDE - intersect_point(2))*N;
		return images.find(PlaneIndex::LEFT)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::RIGHT:
		i = (HALF_SIDE - intersect_point(0))*M;
		j = (HALF_SIDE - intersect_point(2))*N;
		return images.find(PlaneIndex::RIGHT)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::FRONT:
		i = (HALF_SIDE - intersect_point(2))*M;
		j = (HALF_SIDE + intersect_point(1))*N;
		return images.find(PlaneIndex::FRONT)->second.at<cv::Vec3b>(i, j);
		break;

	case PlaneIndex::BACK:
		i = (HALF_SIDE - intersect_point(2))*M;
		j = (HALF_SIDE + intersect_point(1))*N;
		return images.find(PlaneIndex::BACK)->second.at<cv::Vec3b>(i, j);
		break;


	default:
		break;
	}

	return cv::Vec3b{};
}
