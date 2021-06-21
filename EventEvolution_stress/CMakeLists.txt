cmake_minimum_required(VERSION 3.16)

project(EventEvolution_stress)

option(ASSERT "Use assertions" OFF)
option(DEBUG "Switch on all possible assertions" OFF)

execute_process(
    COMMAND python -c "from setuptools_scm import get_version; print(get_version(root='..'))"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE MYVERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "Building ${PROJECT_NAME} ${MYVERSION}")

set(CMAKE_BUILD_TYPE Release)

add_executable(${PROJECT_NAME} main.cpp)

set(HIGHFIVE_USE_BOOST 0)
set(HIGHFIVE_USE_XTENSOR 1)

find_package(FrictionQPotFEM REQUIRED)
find_package(xtensor REQUIRED)
find_package(HighFive REQUIRED)
find_package(docopt REQUIRED)
find_package(fmt REQUIRED)
find_package(cpppath REQUIRED)

target_link_libraries(${PROJECT_NAME}
    FrictionQPotFEM
    xtensor::optimize
    xtensor::use_xsimd
    HighFive
    docopt
    fmt::fmt
    cpppath
    FrictionQPotFEM::compiler_warnings)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DMYVERSION=\"${MYVERSION}\"")

if(DEBUG)
    target_link_libraries(${PROJECT_NAME} FrictionQPotFEM::debug)
elseif(ASSERT)
    target_link_libraries(${PROJECT_NAME} FrictionQPotFEM::assert)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DNDEBUG)
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
