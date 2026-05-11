cmake_minimum_required(VERSION 3.15)

file(GLOB EXPRESS_SOURCES CONFIGURE_DEPENDS
    *.c
)

add_library(aifes_express STATIC ${EXPRESS_SOURCES})

target_include_directories(aifes_express PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_libraries(aifes_express PUBLIC
    aifes_base
    aifes_default
)

set_target_properties(aifes_express PROPERTIES
    LINKER_LANGUAGE CXX
)