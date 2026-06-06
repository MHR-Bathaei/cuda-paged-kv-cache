cmake_minimum_required(VERSION 3.18)
project(paged_kv_cache LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Detect GPU architecture - default to sm_75 (Turing)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_executable(paged_kv_cache
    src/main.cu
 "include/page_table.h" "include/kv_cache_manager.h")

# Crucial: Include both project headers and CUDA Toolkit system headers for IntelliSense
target_include_directories(paged_kv_cache PRIVATE 
    include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)