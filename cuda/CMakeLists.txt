#cuda/CMakeLists.txt

add_subdirectory(global)
add_subdirectory(2D)

add_executable(galaxy_cuda
    galaxy_cuda.cpp
    galaxy_kernel.cu
)

target_include_directories(galaxy_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/glad/include
    ${CMAKE_SOURCE_DIR}/glfw/include
)

find_package(OpenGL REQUIRED)

target_link_libraries(galaxy_cuda PRIVATE
    common
    glfw
    glad
    OpenGL::GL
)

# NO vuelvas a establecer CUDA_ARCHITECTURES aquí
set_target_properties(galaxy_cuda PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
)




