cmake_minimum_required (VERSION 2.8)
project (MicroBenchmark)

option(USE_CLANG "build application with clang" ON) # OFF is the default

SET(CMAKE_CXX_FLAGS "-std=c++17 -O3 -march=native -g3 -ggdb -fno-omit-frame-pointer")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

file(GLOB_RECURSE SRCS src/*/*.cpp src/*/*.h)

add_executable(MicroBenchmark src/run_perf_tests.cpp ${SRCS})
target_link_libraries (MicroBenchmark benchmark pthread)
