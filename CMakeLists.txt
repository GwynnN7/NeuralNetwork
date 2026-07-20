cmake_minimum_required(VERSION 3.14)
project(NeuralNet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()


find_package(Eigen3 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(CLI11)

file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(NeuralNet ${SOURCES})

target_include_directories(NeuralNet PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(NeuralNet PRIVATE Eigen3::Eigen CLI11::CLI11)

target_compile_options(NeuralNet PRIVATE
    # Standard warnings 
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>

    # Optimization and CPU-specific vectorization for Eigen
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3 -march=native>
)