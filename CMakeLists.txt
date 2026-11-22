cmake_minimum_required(VERSION 3.20)
project(parallel_kmeans LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(kmeans
    src/main.cpp
    src/cli.cpp
    src/dataset.cpp
    src/kmeans_common.cpp
    src/kmeans_serial.cpp
)
target_include_directories(kmeans PRIVATE include)

if(MSVC)
    target_compile_options(kmeans PRIVATE /W4 /permissive-)
else()
    target_compile_options(kmeans PRIVATE -Wall -Wextra -Wpedantic)
endif()
