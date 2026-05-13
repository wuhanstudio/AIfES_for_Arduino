set (AIfES_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src)

file(GLOB app_sources
    "./*.c"

    "${AIfES_ROOT_DIR}/*.c"

    "${AIfES_ROOT_DIR}/basic/base/aialgo/*.c"
    "${AIfES_ROOT_DIR}/basic/base/ailayer/*.c"
    "${AIfES_ROOT_DIR}/basic/base/ailoss/*.c"
    "${AIfES_ROOT_DIR}/basic/base/aimath/*.c"
    "${AIfES_ROOT_DIR}/basic/base/aiopti/*.c"

    "${AIfES_ROOT_DIR}/basic/default/ailayer/*.c"
    "${AIfES_ROOT_DIR}/basic/default/ailoss/*.c"
    "${AIfES_ROOT_DIR}/basic/default/aimath/*.c"
    "${AIfES_ROOT_DIR}/basic/default/aiopti/*.c"

    "${AIfES_ROOT_DIR}/basic/express/*.c"
)

idf_component_register(
    SRCS ${app_sources}
    INCLUDE_DIRS 
        "."
        "${AIfES_ROOT_DIR}"
        "${AIfES_ROOT_DIR}/core"
        "${AIfES_ROOT_DIR}/basic/base"
        "${AIfES_ROOT_DIR}/basic/default"
        "${AIfES_ROOT_DIR}/basic/express"
)

target_compile_definitions(${COMPONENT_LIB}
    PRIVATE AIDEBUG_ENABLE_PRINTING
)

target_compile_options(${COMPONENT_LIB} PRIVATE
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