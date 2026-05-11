# src/cnn/CMakeLists.txt

cmake_minimum_required(VERSION 3.15)

#
# CNN Base Layer Implementations
#

file(GLOB_RECURSE AIFES_CNN_BASE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/base/**/*.c
)

add_library(aifes_cnn_base STATIC
    ${AIFES_CNN_BASE_SOURCES}
)

target_include_directories(aifes_cnn_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/base
    ${CMAKE_CURRENT_SOURCE_DIR}/base/ailayer
)

target_link_libraries(aifes_cnn_base PUBLIC
    aifes_base
)

set_target_properties(aifes_cnn_base PROPERTIES
    LINKER_LANGUAGE CXX
)

#
# CNN Default Backend
#

file(GLOB_RECURSE AIFES_CNN_DEFAULT_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/default/**/*.c
)

add_library(aifes_cnn_default STATIC
    ${AIFES_CNN_DEFAULT_SOURCES}
)

target_include_directories(aifes_cnn_default PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/default
    ${CMAKE_CURRENT_SOURCE_DIR}/default/ailayer
    ${CMAKE_CURRENT_SOURCE_DIR}/default/aimath
)

target_link_libraries(aifes_cnn_default PUBLIC
    aifes_cnn_base
    aifes_default
    aifes_base
)

set_target_properties(aifes_cnn_default PROPERTIES
    LINKER_LANGUAGE CXX
)

#
# Common compile definitions
#

target_compile_definitions(aifes_cnn_base PUBLIC
    AIFES_CNN_ENABLE
)

target_compile_definitions(aifes_cnn_default PUBLIC
    AIFES_CNN_ENABLE
)

#
# Optional optimization flags
#

if(NOT MSVC)
    target_compile_options(aifes_cnn_base PRIVATE
        -O3
        -ffast-math
    )

    target_compile_options(aifes_cnn_default PRIVATE
        -O3
        -ffast-math
    )
endif()
