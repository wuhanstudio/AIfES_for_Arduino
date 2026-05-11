cmake_minimum_required(VERSION 3.15)

file(GLOB_RECURSE DEFAULT_SOURCES CONFIGURE_DEPENDS
    ailayer/*.c
    ailoss/*.c
    aimath/*.c
    aiopti/*.c
)

add_library(aifes_default STATIC ${DEFAULT_SOURCES})

target_include_directories(aifes_default PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${CMAKE_CURRENT_SOURCE_DIR}/ailayer
    ${CMAKE_CURRENT_SOURCE_DIR}/ailoss
    ${CMAKE_CURRENT_SOURCE_DIR}/aimath
    ${CMAKE_CURRENT_SOURCE_DIR}/aiopti
)

target_link_libraries(aifes_default PUBLIC
    aifes_base
)

set_target_properties(aifes_default PROPERTIES
    LINKER_LANGUAGE CXX
)