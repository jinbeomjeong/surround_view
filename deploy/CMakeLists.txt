cmake_minimum_required(VERSION 3.30)
project(deploy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(OpenCV_DIR D:/opencv-4.11.0/build/install)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(deploy src/main.cpp src/surround_view.cpp)

target_link_libraries(deploy ${OpenCV_LIBRARIES})


