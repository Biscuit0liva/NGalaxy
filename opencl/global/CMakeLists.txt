# opencl/CMakeLists.txt

find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(galaxy_opencl_global
  galaxy_opencl_global.cpp
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/galaxy_kernel_global.cl
  ${CMAKE_BINARY_DIR}/galaxy_kernel_global.cl
  COPYONLY
)

target_include_directories(galaxy_opencl_global PRIVATE
  ${OpenCL_INCLUDE_DIRS}/common
  ${CMAKE_SOURCE_DIR}/glad/include
  ${CMAKE_SOURCE_DIR}/glfw/include
  ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(galaxy_opencl_global PRIVATE
  common glfw glad OpenCL::OpenCL OpenGL::GL 
)
