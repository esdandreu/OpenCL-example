# Installing RUNTIME_DEPENDENCIES requires at least CMake version 3.21
cmake_minimum_required(VERSION 3.16)

# Compliler flags
set(CMAKE_CXX_STANDARD 17)

# Set CMake to use `vcpkg` to install dependencies. I avoids the need of
# passing it as an argument to `cmake`.
# https://github.com/microsoft/vcpkg/blob/master/docs/users/integration.md#using-an-environment-variable-instead-of-a-command-line-option
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE 
            "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING ""
        )
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
        set(CMAKE_TOOLCHAIN_FILE 
            "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING ""
        )
    endif()
endif()
if (DEFINED CMAKE_TOOLCHAIN_FILE) # Print useful information
    message(STATUS "Using toolchain: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(STATUS "\
Not using `vcpkg`. Consider adding it to manage \
dependencies https://github.com/microsoft/vcpkg.git"
    )
endif()

# In order to avoid specifying package name and version in multiple files, we
# will use `vcpkg.json` in the repository root as reference and extract the
# apropiate variables from there.
file(READ ${CMAKE_SOURCE_DIR}/vcpkg.json vcpkg_json)
string(JSON PROJECT_NAME GET ${vcpkg_json} "name")
string(JSON VERSION_STRING GET ${vcpkg_json} "version-string")

project(${PROJECT_NAME} VERSION ${VERSION_STRING})

# It is recommended to split this file into multiple files and use
# `add_submodule` if it starts to grow too large.

# ------------------------- Build the project library -------------------------
# Recomended refactor into src/${PROJECT_NAME}/CMakeLists.txt

# It is important that the library is static in order to build its Python
# wrapper
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCL REQUIRED)

add_library(${PROJECT_NAME} STATIC 
    src/${PROJECT_NAME}/matmul.cpp
)

target_include_directories(${PROJECT_NAME} 
    PUBLIC
        src/${PROJECT_NAME}
)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE
        Eigen3::Eigen
        OpenCL::OpenCL
)

# ---------------------------- Build library tests ----------------------------
# Recomended refactor into tests/CMakeLists.txt

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    message(STATUS "Build tests")
    find_package(GTest CONFIG REQUIRED)

    enable_testing()

    # Tests can be run with `ctest` or by directly running the test executable
    # that will be named as `${PROJECT_TESTS_NAME}`. The name of the executable
    # is not really relevant as `ctest` will find it anyway thanks to Google
    # Test.
    set(PROJECT_TESTS_NAME test_${PROJECT_NAME})

    file(GLOB TEST_FILES tests/test_*.cpp)
    add_executable(${PROJECT_TESTS_NAME} 
        ${TEST_FILES}
    )

    target_link_libraries(${PROJECT_TESTS_NAME} 
        PRIVATE
            ${PROJECT_NAME}
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(${PROJECT_TESTS_NAME})

endif()

# ------------------------- Build library benchmamrk --------------------------
# Recomended refactor into benchmark/CMakeLists.txt

option(BUILD_BENCHMARK "Build benchmark" ON)
if(BUILD_BENCHMARK)
    message(STATUS "Build benchmark")
    find_package(benchmark CONFIG REQUIRED)

    set(PROJECT_BENCHMARK_NAME benchmark_${PROJECT_NAME})

    file(GLOB BENCHMARK_FILES benchmark/bm_*.cpp)
    add_executable(${PROJECT_BENCHMARK_NAME} 
        ${BENCHMARK_FILES}
    )

    target_link_libraries(${PROJECT_BENCHMARK_NAME} 
        PRIVATE
            ${PROJECT_NAME}
            benchmark::benchmark_main
    )


endif()