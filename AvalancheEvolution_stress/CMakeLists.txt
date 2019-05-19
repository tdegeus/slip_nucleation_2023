cmake_minimum_required(VERSION 3.1)

# basic info
# ----------

# define a project name
project(Run)

# define empty list of libraries to link
set(PROJECT_LIBS "")

option(ASSERT "Use assertions" OFF)
option(OPENMP "Use parallelization" OFF)

# copy files
# ----------

configure_file("makeJob_compile.py" "makeJob_compile.py" COPYONLY)
configure_file("collect.py" "collect.py" COPYONLY)

# basic compiler options
# ----------------------

# set optimization level
set(CMAKE_BUILD_TYPE Release)

# set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# optimisation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DXTENSOR_USE_XSIMD=ON")

# assertions
if(NOT ASSERT)
  add_definitions(-DNDEBUG)
else()
  add_definitions(-DXTENSOR_ENABLE_ASSERT=ON -DGMATELASTOPLASTICQPOT_ENABLE_ASSERT -DGOOSEFEM_ENABLE_ASSERT)
endif()

# warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -g")

# OpenMP settings
if(OPENMP)
  if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    find_package(OpenMP)
    if(OPENMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

# load packages
# -------------

# add custom include paths
if(NOT "$ENV{INCLUDE_PATH}" STREQUAL "")
  string(REPLACE ":" ";" INCLUDE_LIST "$ENV{INCLUDE_PATH}")
  include_directories(${INCLUDE_LIST})
endif()

# load BLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(PROJECT_LIBS ${PROJECT_LIBS} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# load HDF5
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
set(PROJECT_LIBS ${PROJECT_LIBS} ${HDF5_C_LIBRARIES})

# load docopt.cpp
find_package(docopt COMPONENTS CXX REQUIRED)
include_directories(${DOCOPT_INCLUDE_DIRS})

# load eigen
find_package(PkgConfig)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

# load FMT
pkg_check_modules(fmt REQUIRED fmt)
include_directories(SYSTEM ${fmt_INCLUDE_DIRS})
link_directories(SYSTEM ${fmt_LIBRARY_DIRS})
set(PROJECT_LIBS ${PROJECT_LIBS} ${fmt_LIBRARIES})

# create executable
# -----------------

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${PROJECT_LIBS} docopt)

