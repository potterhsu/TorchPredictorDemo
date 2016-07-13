cmake_minimum_required(VERSION 3.2)
project(TorchPredictorDemo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

SET(OPENBLAS_HOME $ENV{OPENBLAS_HOME})
IF(NOT OPENBLAS_HOME)
    MESSAGE(FATAL_ERROR "Environment variable OPENBLAS_HOME is undefined.")
ENDIF()
include_directories("${OPENBLAS_HOME}/include")
link_directories("${OPENBLAS_HOME}/lib")

add_executable(ImageDemo ImageDemo.cpp ModelHelper.hpp)
target_link_libraries(ImageDemo ${OpenCV_LIBS} openblas TorchPredictor)

add_executable(CameraDemo CameraDemo.cpp ModelHelper.hpp)
target_link_libraries(CameraDemo ${OpenCV_LIBS} openblas TorchPredictor)