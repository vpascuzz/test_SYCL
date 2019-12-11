# Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "dpcpp")
cmake_minimum_required( VERSION 3.8 FATAL_ERROR )

project(test_SYCL)

set(CMAKE_CXX_STANDARD 14 CACHE STRING
  "The C++ standard to use of the build")

# list(APPEND CMAKE_MODULE_PATH ${ComputeCpp_SDK_DIR}/cmake/Modules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsycl -std=c++11 -O2 -cl-opt-disable")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -ltbb -lOpenCL -lsycl")

# find_package( ComputeCpp REQUIRED )


add_executable(tbb-task-sycl src/tbb-task-sycl.cpp)
# add_sycl_to_target(TARGET tbb-task-sycl SOURCES src/tbb-task-sycl.cpp)

# TBB dependencies
# target_include_directories(tbb-task-sycl PRIVATE ${TBB_INCLUDE_DIRS})
# target_link_libraries(tbb-task-sycl tbb)
