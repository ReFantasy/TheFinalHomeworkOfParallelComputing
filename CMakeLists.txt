CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT(ViewGeneration)

#字符集
Add_Definitions(-DUNICODE -D_UNICODE)

#ADD_COMPILE_OPTIONS(-std=c++11)
SET(CMAKE_CXX_FLAGS "-std=c++11")

# 配置 CUDA 库
FIND_PACKAGE(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS}) 


# 配置 OpenCV 库
FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
LINK_LIBRARIES( ${OpenCV_LIBS} )

# 配置 第三方包含目录
#include_directories(./3rdparty)

# 编译版本
SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_BUILD_TYPE "Debug")

#添加子目录
ADD_SUBDIRECTORY(src)









