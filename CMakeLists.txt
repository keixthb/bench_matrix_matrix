cmake_minimum_required(VERSION 3.12.0...3.21.0)

project(benchmark VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(p3a_ARCHITECTURES "" CACHE STRING "GPU architectures (either CUDA_ARCHITECTURES or HIP_ARCHITECTURES)")


set(ENABLE_CUDA "${p3a_ENABLE_CUDA}") 
set(ENABLE_HIP "${p3a_ENABLE_HIP}") 

find_package(p3a REQUIRED) 
find_package(GTest REQUIRED) 
find_package(mpicpp REQUIRED)  

set(benchmark_HEADERS
  )

set(BENCHMARK_SOURCES
  main.cpp
  )

add_executable(benchmark ${BENCHMARK_SOURCES})

target_link_libraries(benchmark PUBLIC p3a::p3a)
target_link_libraries(benchmark PUBLIC Kokkos::kokkos)
target_link_libraries(benchmark PUBLIC GTest::GTest)


