cmake_minimum_required(VERSION 3.16)

project(Run)

option(ASSERT "Use assertions" OFF)

# Get the current working branch
execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get the latest commit hash
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_BUILD_TYPE Release)

add_executable(${PROJECT_NAME} main.cpp)

set(HIGHFIVE_USE_BOOST 0)
set(HIGHFIVE_USE_XTENSOR 1)

find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(HighFive REQUIRED)
find_package(docopt REQUIRED)
find_package(fmt REQUIRED)
find_package(cpppath REQUIRED)
find_package(GooseFEM REQUIRED)
find_package(GMatElastoPlasticQPot REQUIRED)

target_link_libraries(${PROJECT_NAME}
    xtensor
    xtensor::optimize
    xtensor::use_xsimd
    xtensor-blas
    HighFive
    docopt
    fmt::fmt
    cpppath
    GooseFEM
    GooseFEM::compiler_warnings
    GMatElastoPlasticQPot)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

if(NOT ASSERT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DNDEBUG)
else()
    target_link_libraries(${PROJECT_NAME}
        GooseFEM::assert
        GMatElastoPlasticQPot::assert)
endif()

if (APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_RPATH ON)
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH 1
        SKIP_BUILD_RPATH FALSE)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
