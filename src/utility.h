#ifndef __UTILITY_H__
#define __UTILITY_H__
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>

enum class PlaneIDA{ UP, DOWN, LEFT, RIGHT, FRONT, BACK };

/*
 * 判断射线 OP 与正方体相交的平面
 * 
 */

PlaneIDA WhichPlane(Eigen::Vector3d pt, Eigen::Vector3d &intersect_point);




#endif//__UTILITY_H__
