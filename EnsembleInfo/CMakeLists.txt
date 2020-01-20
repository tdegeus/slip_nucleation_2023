cmake_minimum_required(VERSION 3.1)

project(EnsembleInfo)

option(ASSERT "Use assertions" OFF)

set(CMAKE_BUILD_TYPE Release)

add_executable(${PROJECT_NAME} main.cpp)

set(HIGHFIVE_USE_BOOST 0)
set(HIGHFIVE_USE_XTENSOR 1)

find_package(xtensor REQUIRED)
find_package(HighFive REQUIRED)
find_package(docopt REQUIRED)
find_package(GooseFEM REQUIRED)
find_package(GMatElastoPlasticQPot REQUIRED)
find_package(fmt REQUIRED)

target_link_libraries(${PROJECT_NAME}
    xtensor
    xtensor::optimize
    xtensor::use_xsimd
    HighFive
    GooseFEM
    GMatElastoPlasticQPot
    docopt
    fmt::fmt)

target_compile_options(${PROJECT_NAME} PRIVATE
    -march=native
    -Wall -Wextra -pedantic -g -Wno-unknown-pragmas)

if(NOT ASSERT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DNDEBUG)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        -DXTENSOR_ENABLE_ASSERT=ON
        -DGMATELASTOPLASTICQPOT_ENABLE_ASSERT
        -DGOOSEFEM_ENABLE_ASSERT)
endif()
