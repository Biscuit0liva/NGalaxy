#cuda/CMakeLists.txt

add_executable(galaxy_cuda_2D
    galaxy_cuda_2D.cpp
    galaxy_kernel_2D.cu
)

target_include_directories(galaxy_cuda_2D PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/glad/include
    ${CMAKE_SOURCE_DIR}/glfw/include
)

find_package(OpenGL REQUIRED)

target_link_libraries(galaxy_cuda_2D PRIVATE
    common
    glfw
    glad
    OpenGL::GL
)

# NO vuelvas a establecer CUDA_ARCHITECTURES aquí
set_target_properties(galaxy_cuda_2D PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)




