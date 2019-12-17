#ifndef __UTILITY_H__
#define __UTILITY_H__
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>
enum class PlaneIndex{ UP, BOTTOM, LEFT, RIGHT, FRONT, BACK,NONE };

/*
 * 判断射线 OP 与正方体相交的平面
 * 
 */

PlaneIndex WhichPlane(Eigen::Vector3d pt, Eigen::Vector3d &intersect_point);

// 获取穿过正方体的像素颜色
const cv::Vec3b GetPixel(const std::map<PlaneIndex, cv::Mat> &images, Eigen::Vector3d pt, int M, int N);


#endif//__UTILITY_H__
