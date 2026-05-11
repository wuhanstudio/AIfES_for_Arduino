cmake_minimum_required(VERSION 3.15)

project(AIfES_Linux LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)

add_definitions( -DAIDEBUG_ENABLE_PRINTING)

#
# Enable warnings
#

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(
        -Wall
        -Wextra
        -Wno-pedantic
        -Wno-unused-parameter
        -Wno-maybe-uninitialized
        -Wno-sign-compare
        -Wno-missing-field-initializers
        -Wno-absolute-value
    )
endif()

set (APP_MAIN aifes_mnist_training)
set (AIfES_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../src)
#
# Add modules
#

set(AIfES_BASE_BINARY_DIR ${CMAKE_BINARY_DIR}/basic/base)
set(AIfES_DEFAULT_BINARY_DIR ${CMAKE_BINARY_DIR}/basic/default)
set(AIfES_EXPRESS_BINARY_DIR ${CMAKE_BINARY_DIR}/basic/express)
set(AIfES_CNN_BINARY_DIR ${CMAKE_BINARY_DIR}/cnn)

add_subdirectory(${AIfES_ROOT_DIR}/basic/base ${AIfES_BASE_BINARY_DIR})
add_subdirectory(${AIfES_ROOT_DIR}/basic/default ${AIfES_DEFAULT_BINARY_DIR})
add_subdirectory(${AIfES_ROOT_DIR}/basic/express ${AIfES_EXPRESS_BINARY_DIR})
add_subdirectory(${AIfES_ROOT_DIR}/cnn ${AIfES_CNN_BINARY_DIR})

#
# Executable
#

add_executable(${APP_MAIN}
    main.c
    ${AIfES_ROOT_DIR}/aifes_config.c
)

#
# Include paths
#

target_include_directories(${APP_MAIN} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIfES_ROOT_DIR}/core
    ${AIfES_ROOT_DIR}/basic/base
    ${AIfES_ROOT_DIR}/basic/default
    ${AIfES_ROOT_DIR}/basic/express
)

#
# Link libraries
#

target_link_libraries(${APP_MAIN} PRIVATE
    aifes_base
    aifes_default
    aifes_express
    aifes_cnn_base
    aifes_cnn_default
)

#
# Linux math library
#

target_link_libraries(${APP_MAIN} PRIVATE
    m
)

configure_file(${CMAKE_SOURCE_DIR}/mnist/train-images-idx3-ubyte
               ${CMAKE_BINARY_DIR}/mnist/train-images-idx3-ubyte
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/mnist/train-labels-idx1-ubyte
               ${CMAKE_BINARY_DIR}/mnist/train-labels-idx1-ubyte
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/mnist/t10k-images-idx3-ubyte
               ${CMAKE_BINARY_DIR}/mnist/t10k-images-idx3-ubyte
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/mnist/t10k-labels-idx1-ubyte
               ${CMAKE_BINARY_DIR}/mnist/t10k-labels-idx1-ubyte
               COPYONLY)
