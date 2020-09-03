cmake_minimum_required(VERSION 3.16)

project(EventEvolution_stress)

option(ASSERT "Use assertions" OFF)

set(CMAKE_BUILD_TYPE Release)

add_executable(${PROJECT_NAME} main.cpp)

set(HIGHFIVE_USE_BOOST 0)
set(HIGHFIVE_USE_XTENSOR 1)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(xtensor REQUIRED)
find_package(HighFive REQUIRED)
find_package(docopt REQUIRED)
find_package(GooseFEM REQUIRED)
find_package(GMatElastoPlasticQPot REQUIRED)
find_package(fmt REQUIRED)

target_link_libraries(${PROJECT_NAME}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
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

if (APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        MACOSX_RPATH ON
    )
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH 1
        SKIP_BUILD_RPATH FALSE
    )
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
