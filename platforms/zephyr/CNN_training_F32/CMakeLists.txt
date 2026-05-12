cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(aifes_cnn_training)

set (AIfES_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../src)

zephyr_compile_options(
    -Wall
    -Wextra
    -Wno-pedantic
    -Wno-unused-parameter
    -Wno-maybe-uninitialized
    -Wno-sign-compare
    -Wno-missing-field-initializers
    -Wno-absolute-value
    -Wno-pointer-arith
    -Wno-double-promotion
)

target_compile_definitions(app
    PRIVATE AIDEBUG_ENABLE_PRINTING
)

target_include_directories(app PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIfES_ROOT_DIR}
    ${AIfES_ROOT_DIR}/core
    ${AIfES_ROOT_DIR}/basic/base
    ${AIfES_ROOT_DIR}/basic/default
    ${AIfES_ROOT_DIR}/basic/express
)

file(GLOB app_sources 
    src/* 
    ${AIfES_ROOT_DIR}/*
    ${AIfES_ROOT_DIR}/basic/base/aialgo/*
    ${AIfES_ROOT_DIR}/basic/base/ailayer/*
    ${AIfES_ROOT_DIR}/basic/base/ailoss/* 
    ${AIfES_ROOT_DIR}/basic/base/aimath/* 
    ${AIfES_ROOT_DIR}/basic/base/aiopti/* 
    ${AIfES_ROOT_DIR}/basic/default/ailayer/*
    ${AIfES_ROOT_DIR}/basic/default/ailoss/*
    ${AIfES_ROOT_DIR}/basic/default/aimath/*
    ${AIfES_ROOT_DIR}/basic/default/aiopti/*
    ${AIfES_ROOT_DIR}/basic/express/*
    ${AIfES_ROOT_DIR}/cnn/base/ailayer/*
    ${AIfES_ROOT_DIR}/cnn/default/ailayer/*
    ${AIfES_ROOT_DIR}/cnn/default/aimath/*
)

target_sources(app PRIVATE ${app_sources})
