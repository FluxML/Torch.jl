cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(doeye_caml)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR};${CMAKE_MODULE_PATH}")

find_package(Torch REQUIRED)

option(TORCH_C_API_CUDA "Use CUDA" OFF)
if(TORCH_C_API_CUDA)
    enable_language(CUDA)
endif()

include_directories(SYSTEM path)

add_library(doeye_caml "SHARED" torch_api.cpp)
target_link_libraries(doeye_caml "${TORCH_LIBRARIES}")

set_property(TARGET doeye_caml PROPERTY CXX_STANDARD 14)
