cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(st_hosvd_research)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)

set(LIB_SOURCES
    src/st_hosvd.cpp
    src/qr_st_hosvd.cpp
    src/qr_decomposition.cpp
)

# Main Tensor Decomp
add_executable(tensor_decomp src/main.cpp ${LIB_SOURCES})
target_link_libraries(tensor_decomp "${TORCH_LIBRARIES}")

# Original Performance Benchmark
add_executable(benchmark src/benchmark.cpp ${LIB_SOURCES})
target_link_libraries(benchmark "${TORCH_LIBRARIES}")

# QR Algorithm Memory Benchmark
add_executable(bench_qr_memory src/bench_qr_memory.cpp ${LIB_SOURCES})
target_link_libraries(bench_qr_memory "${TORCH_LIBRARIES}")

# ST-HOSVD Reconstruction Stability
add_executable(bench_hosvd_stability src/bench_hosvd_stability.cpp ${LIB_SOURCES})
target_link_libraries(bench_hosvd_stability "${TORCH_LIBRARIES}")

# Least-Squares Gram vs QR
add_executable(ls_stability src/ls_stability.cpp)
target_link_libraries(ls_stability "${TORCH_LIBRARIES}")
