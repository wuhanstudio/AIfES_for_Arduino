/*
  AIfES CNN Training F32 Example (Linux C Version)
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "aifes.h"

int main(void)
{
    uint32_t i;

    printf("AIfES CNN Training F32 Example\n\n");

    float input_data[] = {
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,

        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,

        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,

        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f
    };

    uint16_t input_shape[] = {4, 1, 4, 3};
    aitensor_t input_tensor =
        AITENSOR_4D_F32(input_shape, input_data);

    float target_data[] = {
        0.0f,
        1.0f,
        0.0f,
        1.0f
    };

    uint16_t target_shape[] = {4, 1};
    aitensor_t target_tensor =
        AITENSOR_2D_F32(target_shape, target_data);

    float test_data[] = {
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,

        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,

        1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,

        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.0f
    };

    uint16_t test_shape[] = {4, 1, 4, 3};
    aitensor_t test_tensor =
        AITENSOR_4D_F32(test_shape, test_data);

    float test_target_data[] = {
        0.0f,
        0.0f,
        1.0f,
        1.0f
    };

    uint16_t test_target_shape[] = {4, 1};
    aitensor_t test_target_tensor =
        AITENSOR_2D_F32(test_target_shape, test_target_data);

    //
    // Layer definition
    //

    uint16_t input_layer_shape[] = {4, 1, 4, 3};

    ailayer_input_f32_t input_layer =
        AILAYER_INPUT_F32_A(4, input_layer_shape);

    ailayer_conv2d_t conv2d_layer_1 =
        AILAYER_CONV2D_F32_A(
            2,
            HW(2, 2),
            HW(2, 1),
            HW(1, 1),
            HW(1, 1));

    ailayer_batch_norm_f32_t bn_layer_1 =
        AILAYER_BATCH_NORM_F32_A(0.9f, 1e-6f);

    ailayer_relu_f32_t relu_layer_1 =
        AILAYER_RELU_F32_A();

    ailayer_maxpool2d_t maxpool2d_layer_1 =
        AILAYER_MAXPOOL2D_F32_A(
            HW(2, 1),
            HW(2, 1),
            HW(0, 0));

    ailayer_conv2d_t conv2d_layer_2 =
        AILAYER_CONV2D_F32_A(
            1,
            HW(1, 1),
            HW(1, 1),
            HW(1, 1),
            HW(0, 0));

    ailayer_batch_norm_f32_t bn_layer_2 =
        AILAYER_BATCH_NORM_F32_A(0.9f, 1e-6f);

    ailayer_relu_f32_t relu_layer_2 =
        AILAYER_RELU_F32_A();

    ailayer_flatten_t flatten_layer =
        AILAYER_FLATTEN_F32_A();

    ailayer_dense_f32_t dense_layer_1 =
        AILAYER_DENSE_F32_A(1);

    ailayer_sigmoid_f32_t sigmoid_layer_3 =
        AILAYER_SIGMOID_F32_A();

    ailoss_mse_f32_t mse_loss;

    //
    // Model structure
    //

    aimodel_t model;
    ailayer_t *x;

    model.input_layer =
        ailayer_input_f32_default(&input_layer);

    x = ailayer_conv2d_chw_f32_default(
        &conv2d_layer_1,
        model.input_layer);

    x = ailayer_batch_norm_cfirst_f32_default(
        &bn_layer_1,
        x);

    x = ailayer_relu_f32_default(
        &relu_layer_1,
        x);

    x = ailayer_maxpool2d_chw_f32_default(
        &maxpool2d_layer_1,
        x);

    x = ailayer_conv2d_chw_f32_default(
        &conv2d_layer_2,
        x);

    x = ailayer_batch_norm_cfirst_f32_default(
        &bn_layer_2,
        x);

    x = ailayer_relu_f32_default(
        &relu_layer_2,
        x);

    x = ailayer_flatten_f32_default(
        &flatten_layer,
        x);

    x = ailayer_dense_f32_default(
        &dense_layer_1,
        x);

    x = ailayer_sigmoid_f32_default(
        &sigmoid_layer_3,
        x);

    model.output_layer = x;

    model.loss =
        ailoss_mse_f32_default(
            &mse_loss,
            model.output_layer);

    aialgo_compile_model(&model);

    //
    // Print structure
    //

    printf("-------------- Model structure ---------------\n");
    aialgo_print_model_structure(&model);
    printf("----------------------------------------------\n\n");

    //
    // Parameter memory
    //

    uint32_t parameter_memory_size =
        aialgo_sizeof_parameter_memory(&model);

    printf(
        "Required parameter memory: %u bytes\n",
        parameter_memory_size);

    void *parameter_memory =
        malloc(parameter_memory_size);

    if (parameter_memory == NULL) {
        printf("Parameter memory allocation failed\n");
        return 1;
    }

    aialgo_distribute_parameter_memory(
        &model,
        parameter_memory,
        parameter_memory_size);

    //
    // Initialize parameters
    //

    srand((unsigned int)time(NULL));

    aialgo_initialize_parameters_model(&model);

    //
    // Optimizer
    //

    aiopti_adam_f32_t adam_opti =
        AIOPTI_ADAM_F32(
            0.01f,
            0.9f,
            0.999f,
            1e-6f);

    aiopti_t *optimizer =
        aiopti_adam_f32_default(&adam_opti);

    //
    // Working memory
    //

    uint32_t working_memory_size =
        aialgo_sizeof_training_memory(
            &model,
            optimizer);

    printf(
        "Required training memory: %u bytes\n",
        working_memory_size);

    void *working_memory =
        malloc(working_memory_size);

    if (working_memory == NULL) {
        printf("Working memory allocation failed\n");
        free(parameter_memory);
        return 1;
    }

    aialgo_schedule_training_memory(
        &model,
        optimizer,
        working_memory,
        working_memory_size);

    aialgo_init_model_for_training(
        &model,
        optimizer);

    //
    // Training
    //

    float loss;
    uint32_t batch_size = 4;
    uint32_t epochs = 250;

    printf("\nStart training\n");

    for (i = 0; i < epochs; i++) {

        aialgo_train_model(
            &model,
            &input_tensor,
            &target_tensor,
            optimizer,
            batch_size);

        if (i % 10 == 0) {

            aialgo_calc_loss_model_f32(
                &model,
                &input_tensor,
                &target_tensor,
                &loss);

            printf(
                "Epoch %u: loss = %f\n",
                i,
                loss);
        }
    }

    //
    // Evaluation
    //

    float output_data[4];
    uint16_t output_shape[] = {4, 1};

    aitensor_t output_tensor =
        AITENSOR_2D_F32(
            output_shape,
            output_data);

    printf("\n+++ Training data +++\n");

    aialgo_inference_model(
        &model,
        &input_tensor,
        &output_tensor);

    printf("Predicted values:\n");
    print_aitensor(&output_tensor);

    printf("Target values:\n");
    print_aitensor(&target_tensor);

    printf("\n+++ Test data +++\n");

    aialgo_inference_model(
        &model,
        &test_tensor,
        &output_tensor);

    printf("Predicted values:\n");
    print_aitensor(&output_tensor);

    printf("Target values:\n");
    print_aitensor(&test_target_tensor);

    //
    // Cleanup
    //

    free(working_memory);
    free(parameter_memory);

    return 0;
}