# CMakeLists.txt
cmake_minimum_required(VERSION 3.24)
project(galaxy LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

# Interpolabilidad para NVIDIA en Linux
if(UNIX AND NOT APPLE)
    find_package(CUDAToolkit REQUIRED)
    
    add_compile_definitions(
        __GL_SYNC_TO_VBLANK=0
        __GLX_VENDOR_LIBRARY_NAME=nvidia
    )
    
    add_compile_options(-Wno-deprecated-declarations)
endif()

option(USE_OPENCL "Build with OpenCL instead of CUDA" OFF)

add_subdirectory(glfw)
add_subdirectory(glad)
add_subdirectory(common)

if (USE_OPENCL)
    add_subdirectory(opencl)
else()
    add_subdirectory(cuda)
endif()
add_subdirectory(sequential)






