#ifndef __UTILITY_H__
#define __UTILITY_H__
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>
enum class PlaneIndex{ UP, BOTTOM, LEFT, RIGHT, FRONT, BACK,NONE };

/*
 * �ж����� OP ���������ཻ��ƽ��
 * 
 */

PlaneIndex WhichPlane(Eigen::Vector3d pt, Eigen::Vector3d &intersect_point);

// ��ȡ�����������������ɫ
const cv::Vec3b GetPixel(const std::map<PlaneIndex, cv::Mat> &images, Eigen::Vector3d pt, int M, int N);


#endif//__UTILITY_H__
