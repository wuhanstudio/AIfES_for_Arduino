cmake_minimum_required(VERSION 3.15)

file(GLOB_RECURSE BASE_SOURCES CONFIGURE_DEPENDS
    aialgo/*.c
    ailayer/*.c
    ailoss/*.c
    aimath/*.c
    aiopti/*.c
)

add_library(aifes_base STATIC ${BASE_SOURCES})

target_include_directories(aifes_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${CMAKE_CURRENT_SOURCE_DIR}/aialgo
    ${CMAKE_CURRENT_SOURCE_DIR}/ailayer
    ${CMAKE_CURRENT_SOURCE_DIR}/ailoss
    ${CMAKE_CURRENT_SOURCE_DIR}/aimath
    ${CMAKE_CURRENT_SOURCE_DIR}/aiopti
)

# Ensure C files compile correctly in C++ projects
set_target_properties(aifes_base PROPERTIES
    LINKER_LANGUAGE CXX
)