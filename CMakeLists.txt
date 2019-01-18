cmake_minimum_required(VERSION 2.8.12)
project(rkpm-rk4)

set(CMAKE_CXX_STANDARD 17)
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Location of Eigen3 header files")

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
message("Build type set to ${CMAKE_BUILD_TYPE}. "
  "Note that Eigen is significantly fasters in Release mode!")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

enable_testing()
add_subdirectory(src)
add_subdirectory(benchmarks)
