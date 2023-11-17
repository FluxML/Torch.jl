cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_c_api)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR};${CMAKE_MODULE_PATH}")

find_package(Torch REQUIRED)

include_directories(SYSTEM path)

add_library(torch_c_api "SHARED" torch_api.cpp)
target_link_libraries(torch_c_api "${TORCH_LIBRARIES}")

set_property(TARGET torch_c_api PROPERTY CXX_STANDARD 14)
