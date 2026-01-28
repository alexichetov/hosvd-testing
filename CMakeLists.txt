cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(st_hosvd_research)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src)

set(LIB_SOURCES src/st_hosvd.cpp src/qr_st_hosvd.cpp)

add_executable(tensor_decomp src/main.cpp ${LIB_SOURCES})
target_link_libraries(tensor_decomp "${TORCH_LIBRARIES}")

add_executable(benchmark src/benchmark.cpp ${LIB_SOURCES})
target_link_libraries(benchmark "${TORCH_LIBRARIES}")

# --- NEW: Least Squares Stability Test ---
add_executable(ls_stability src/ls_stability.cpp)
target_link_libraries(ls_stability "${TORCH_LIBRARIES}")
