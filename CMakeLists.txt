cmake_minimum_required(VERSION 3.24)
project(galaxy LANGUAGES C CXX CUDA)

add_definitions(-D_USE_MATH_DEFINES)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

set(THIRD_PARTY_INCLUDE_DIRECTORIES 
	"${CMAKE_CURRENT_SOURCE_DIR}/glad/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/glfw/include")
set(THIRD_PARTY_LIBRARIES glfw glad)

add_subdirectory(glfw)
add_subdirectory(glad)

# Añade el ejecutable
add_executable(galaxy
    galaxy.cpp
    galaxy_kernel.cu
)

target_link_libraries(galaxy PRIVATE ${THIRD_PARTY_LIBRARIES})
target_include_directories(galaxy PRIVATE ${THIRD_PARTY_INCLUDE_DIRECTORIES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


