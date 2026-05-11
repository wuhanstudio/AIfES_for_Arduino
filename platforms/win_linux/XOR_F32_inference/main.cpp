#include <iostream>
#include <cstdlib>

extern "C" {
#include "aifes.h"
}

int main()
{
    std::cout << "AIfES XOR inference demo (Linux)" << std::endl;

    //
    // Input tensor
    //

    float input_data[] = {0.0f, 1.0f};

    uint16_t input_shape[] = {1, 2};

    aitensor_t input_tensor =
        AITENSOR_2D_F32(input_shape, input_data);

    //
    // Input layer
    //

    uint16_t input_layer_shape[] = {1, 2};

    ailayer_input_f32_t input_layer =
        AILAYER_INPUT_F32_M(
            2,
            input_layer_shape);

    //
    // Dense layer 1
    //

    float weights_data_dense_1[] = {
        -10.1164f,
        -8.4212f,
         5.4396f,
         7.2970f,
        -7.6482f,
        -9.0155f
    };

    float bias_data_dense_1[] = {
        -2.9653f,
         2.3677f,
        -1.5968f
    };

    ailayer_dense_f32_t dense_layer_1 =
        AILAYER_DENSE_F32_M(
            3,
            weights_data_dense_1,
            bias_data_dense_1);

    //
    // Sigmoid 1
    //

    ailayer_sigmoid_f32_t sigmoid_layer_1 =
        AILAYER_SIGMOID_F32_M();

    //
    // Dense layer 2
    //

    float weights_data_dense_2[] = {
        12.0305f,
        -6.5858f,
        11.9371f
    };

    float bias_data_dense_2[] = {
        -5.4247f
    };

    ailayer_dense_f32_t dense_layer_2 =
        AILAYER_DENSE_F32_M(
            1,
            weights_data_dense_2,
            bias_data_dense_2);

    //
    // Sigmoid 2
    //

    ailayer_sigmoid_f32_t sigmoid_layer_2 =
        AILAYER_SIGMOID_F32_M();

    //
    // Model definition
    //

    aimodel_t model;
    ailayer_t* x;

    model.input_layer =
        ailayer_input_f32_default(&input_layer);

    x = ailayer_dense_f32_default(
        &dense_layer_1,
        model.input_layer);

    x = ailayer_sigmoid_f32_default(
        &sigmoid_layer_1,
        x);

    x = ailayer_dense_f32_default(
        &dense_layer_2,
        x);

    model.output_layer =
        ailayer_sigmoid_f32_default(
            &sigmoid_layer_2,
            x);

    //
    // Compile model
    //

    aialgo_compile_model(&model);

    //
    // Print structure
    //

    std::cout << "\nModel structure:\n";
    aialgo_print_model_structure(&model);

    //
    // Allocate inference memory
    //

    uint32_t memory_size =
        aialgo_sizeof_inference_memory(&model);

    std::cout << "\nInference memory: "
              << memory_size
              << " bytes\n";

    uint8_t* memory_ptr =
        static_cast<uint8_t*>(malloc(memory_size));

    aialgo_schedule_inference_memory(
        &model,
        memory_ptr,
        memory_size);

    //
    // Output tensor
    //

    uint16_t output_shape[] = {1, 1};

    float output_data[1];

    aitensor_t output_tensor =
        AITENSOR_2D_F32(output_shape, output_data);

    //
    // Run inference
    //

    aialgo_inference_model(
        &model,
        &input_tensor,
        &output_tensor);

    //
    // Print result
    //

    std::cout << "\nResults:\n";

    std::cout
        << "Input 1: "
        << input_data[0]
        << "\nInput 2: "
        << input_data[1]
        << "\nOutput  : "
        << ((float*)output_tensor.data)[0]
        << "\n";

    free(memory_ptr);

    return 0;
}