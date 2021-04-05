cmake_minimum_required(VERSION 3.16)

project(Evolution)

option(PushBarrier "Compile PushBarrier" ON)
option(PushElement "Compile PushElement" ON)

if (PushBarrier)
    add_subdirectory(PushBarrier)
endif()

if (PushElement)
    add_subdirectory(PushElement)
endif()
