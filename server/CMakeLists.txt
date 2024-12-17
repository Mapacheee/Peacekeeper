cmake_minimum_required(VERSION 3.10)
project(PeacekeeperServer)

set(CMAKE_CXX_STANDARD 14)

set(TENSORRT_LIB /ruta/a/TensorRT/lib)
set(TENSORRT_INCLUDE /ruta/a/TensorRT/include)
find_package(OpenCV REQUIRED)

include_directories(${TENSORRT_INCLUDE})
link_directories(${TENSORRT_LIB})

add_executable(TensorRTServer main.cpp)
target_link_libraries(TensorRTServer nvinfer nvonnxparser cudart ${OpenCV_LIBS})

