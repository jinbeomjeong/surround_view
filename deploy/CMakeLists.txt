cmake_minimum_required(VERSION 3.30)
project(deploy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(OpenCV_DIR C:/Users/JinBe/Downloads/opencv-4.11.0/build/install)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_DIR}/include)

add_executable(deploy main.cpp utils/surround_view.cpp)

target_link_libraries(deploy ${OpenCV_LIBRARIES})


