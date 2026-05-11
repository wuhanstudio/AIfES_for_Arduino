cmake_minimum_required(VERSION 3.15)

project(aifes LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(base)
add_subdirectory(default)
add_subdirectory(cmsis)
add_subdirectory(avr_pgm)
add_subdirectory(express)
