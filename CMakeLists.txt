project(PARTICLE_FILTER)
cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)
set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")
include(ExternalProject)
find_package(Git REQUIRED)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/ext)

ExternalProject_Add(
    uws
    GIT_REPOSITORY https://github.com/uWebSockets/uWebSockets.git
    CONFIGURE_COMMAND "pwd"
    BUILD_IN_SOURCE 1
    BUILD_COMMAND make
    INSTALL_COMMAND ""
    )

ExternalProject_Get_Property(uws install_dir)
include_directories(${install_dir}/src)

include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)
set(sources src/particle_filter.cpp src/main.cpp)

add_executable(particle_filter ${sources})
add_dependencies(particle_filter uws)

target_link_libraries(particle_filter z ssl uv uWS)
