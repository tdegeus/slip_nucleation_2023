cmake_minimum_required(VERSION 3.16)

project(Evolution)

option(PushWeakest "Compile PushWeakest" ON)
option(PushElement "Compile PushElement" ON)

if (PushWeakest)
    add_subdirectory(PushWeakest)
endif()

if (PushElement)
    add_subdirectory(PushElement)
endif()
