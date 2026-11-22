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

# Keep the serial baseline buildable on systems without MPI. The MPI executable
# is added only when a C++ MPI implementation is available.
option(BUILD_MPI "Build the distributed-memory MPI implementation" ON)
if(BUILD_MPI)
    find_package(MPI QUIET COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        add_executable(kmeans_mpi
            src/main_mpi.cpp
            src/cli.cpp
            src/dataset.cpp
            src/kmeans_common.cpp
            src/kmeans_mpi.cpp
        )
        target_include_directories(kmeans_mpi PRIVATE include)
        target_link_libraries(kmeans_mpi PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
    else()
        message(WARNING "MPI C++ not found; kmeans_mpi will not be built. The serial kmeans target is still available.")
    endif()
endif()

if(MSVC)
    target_compile_options(kmeans PRIVATE /W4 /permissive-)
else()
    target_compile_options(kmeans PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TARGET kmeans_mpi)
    if(MSVC)
        target_compile_options(kmeans_mpi PRIVATE /W4 /permissive-)
    else()
        target_compile_options(kmeans_mpi PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()
