# CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(galaxy LANGUAGES C CXX CUDA)

option(USE_OPENCL "Build with OpenCL instead of CUDA" OFF)

add_subdirectory(glfw)
add_subdirectory(glad)
add_subdirectory(common)

if (USE_OPENCL)
  add_subdirectory(opencl)
else()
  add_subdirectory(cuda)
endif()






