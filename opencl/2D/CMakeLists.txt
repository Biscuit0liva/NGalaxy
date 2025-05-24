# opencl/CMakeLists.txt

find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(galaxy_opencl_2D
  galaxy_opencl_2D.cpp
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/galaxy_kernel_2D.cl
  ${CMAKE_BINARY_DIR}/galaxy_kernel_2D.cl
  COPYONLY
)

target_include_directories(galaxy_opencl_2D PRIVATE
  ${OpenCL_INCLUDE_DIRS}/common
  ${CMAKE_SOURCE_DIR}/glad/include
  ${CMAKE_SOURCE_DIR}/glfw/include
  ${OpenCL_INCLUDE_DIRS}
)

target_link_libraries(galaxy_opencl_2D PRIVATE
  common glfw glad OpenCL::OpenCL OpenGL::GL 
)
