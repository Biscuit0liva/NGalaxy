#sequential/CMakeLists.txt

add_executable(galaxy_sequential
    galaxy_sequential.cpp
)

target_include_directories(galaxy_sequential PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/glad/include
    ${CMAKE_SOURCE_DIR}/glfw/include
)

find_package(OpenGL REQUIRED)

target_link_libraries(galaxy_sequential PRIVATE
    common
    glfw
    glad
    OpenGL::GL
)





