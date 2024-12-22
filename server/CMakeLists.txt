cmake_minimum_required(VERSION 3.10)
project(Peacekeeper)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(Peacekeeper main.cpp)
target_link_libraries(IvyVL ${CUDA_LIBRARIES} nvinfer)