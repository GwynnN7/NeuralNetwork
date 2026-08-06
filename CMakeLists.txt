cmake_minimum_required(VERSION 3.18)
project(NeuralNet LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(NN_USE_BLAS "Use an external BLAS library for matrix operations" ON)
option(NN_NATIVE_ARCH "Build with -march=native (non-portable binary)" ON)
option(NN_DOUBLE_PRECISION "Use double instead of float for Scalar" OFF)
option(NN_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.6.2
)
FetchContent_MakeAvailable(CLI11)

file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(NeuralNet ${SOURCES})
target_include_directories(NeuralNet PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(NeuralNet PRIVATE Eigen3::Eigen CLI11::CLI11 OpenMP::OpenMP_CXX)

if(NN_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_link_libraries(NeuralNet PRIVATE BLAS::BLAS)
    target_compile_definitions(NeuralNet PRIVATE EIGEN_USE_BLAS)
endif()

if(NN_DOUBLE_PRECISION)
    target_compile_definitions(NeuralNet PRIVATE SCALAR_TYPE=double)
endif()

target_compile_options(NeuralNet PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>,$<BOOL:${NN_NATIVE_ARCH}>>:-march=native>
)

if(NN_SANITIZE)
    target_compile_options(NeuralNet PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(NeuralNet PRIVATE -fsanitize=address,undefined)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported AND NOT NN_SANITIZE)
    set_target_properties(NeuralNet PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
