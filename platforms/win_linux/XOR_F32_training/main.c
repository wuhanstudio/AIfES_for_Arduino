#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "aifes.h"

int main()
{
    char cmd[64];

    printf("AIfES XOR training demo (Linux C)\n");
    printf("Type 'training' to start\n");

    // Seed RNG (Linux replacement for Arduino analog noise)
    srand((unsigned)time(NULL));

    while (1)
    {
        printf("\n> ");
        fflush(stdout);

        if (fgets(cmd, sizeof(cmd), stdin) == NULL)
            continue;

        // remove newline
        cmd[strcspn(cmd, "\n")] = 0;

        if (strstr(cmd, "training") == NULL)
        {
            printf("unknown\n");
            continue;
        }

        printf("\nAIfES:\n");
        printf("rand test: %d\n\n", rand());

        //
        // ---------------- INPUT DATA ----------------
        //

        uint16_t input_shape[] = {4, 2};

        float input_data[8] = {
            0.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f
        };

        aitensor_t input_tensor =
            AITENSOR_2D_F32(input_shape, input_data);

        //
        // ---------------- TARGET DATA ----------------
        //

        uint16_t target_shape[] = {4, 1};

        float target_data[4] = {
            0.0f,
            1.0f,
            1.0f,
            0.0f
        };

        aitensor_t target_tensor =
            AITENSOR_2D_F32(target_shape, target_data);

        //
        // ---------------- OUTPUT ----------------
        //

        uint16_t output_shape[] = {4, 1};
        float output_data[4] = {0};

        aitensor_t output_tensor =
            AITENSOR_2D_F32(output_shape, output_data);

        //
        // ---------------- LAYERS ----------------
        //

        uint16_t input_layer_shape[] = {1, 2};

        ailayer_input_f32_t input_layer =
            AILAYER_INPUT_F32_A(2, input_layer_shape);

        ailayer_dense_f32_t dense1 =
            AILAYER_DENSE_F32_A(3);

        ailayer_sigmoid_f32_t sig1 =
            AILAYER_SIGMOID_F32_A();

        ailayer_dense_f32_t dense2 =
            AILAYER_DENSE_F32_A(1);

        ailayer_sigmoid_f32_t sig2 =
            AILAYER_SIGMOID_F32_A();

        ailoss_mse_t mse_loss;

        //
        // ---------------- MODEL ----------------
        //

        aimodel_t model;
        ailayer_t *x;

        model.input_layer =
            ailayer_input_f32_default(&input_layer);

        x = ailayer_dense_f32_default(&dense1, model.input_layer);
        x = ailayer_sigmoid_f32_default(&sig1, x);

        x = ailayer_dense_f32_default(&dense2, x);
        x = ailayer_sigmoid_f32_default(&sig2, x);

        model.output_layer = x;

        model.loss =
            ailoss_mse_f32_default(&mse_loss, model.output_layer);

        aialgo_compile_model(&model);

        printf("\n--- Model structure ---\n");
        aialgo_print_model_structure(&model);
        aialgo_print_loss_specs(model.loss);
        printf("\n-----------------------\n");

        //
        // ---------------- PARAM MEMORY ----------------
        //

        uint32_t param_size =
            aialgo_sizeof_parameter_memory(&model);

        printf("Parameter memory: %u bytes\n", param_size);

        uint8_t *param_mem =
            (uint8_t *)malloc(param_size);

        aialgo_distribute_parameter_memory(
            &model,
            param_mem,
            param_size);

        //
        // ---------------- INIT WEIGHTS ----------------
        //

        aimath_f32_default_init_glorot_uniform(&dense1.weights);
        aimath_f32_default_init_zeros(&dense1.bias);

        aimath_f32_default_init_glorot_uniform(&dense2.weights);
        aimath_f32_default_init_zeros(&dense2.bias);

        //
        // ---------------- OPTIMIZER ----------------
        //

        aiopti_adam_f32_t adam =
            AIOPTI_ADAM_F32(0.1f, 0.9f, 0.999f, 1e-7f);

        aiopti_t *opt =
            aiopti_adam_f32_default(&adam);

        //
        // ---------------- TRAIN MEMORY ----------------
        //

        uint32_t mem_size =
            aialgo_sizeof_training_memory(&model, opt);

        printf("Training memory: %u bytes\n", mem_size);

        uint8_t *mem =
            (uint8_t *)malloc(mem_size);

        if (!mem)
        {
            printf("ERROR: not enough memory\n");
            return 1;
        }

        aialgo_schedule_training_memory(
            &model, opt, mem, mem_size);

        aialgo_init_model_for_training(&model, opt);

        //
        // ---------------- BEFORE TRAIN ----------------
        //

        aialgo_inference_model(
            &model,
            &input_tensor,
            &output_tensor);

        printf("\nBefore training:\n");
        printf("x1\tx2\ty\ty_pred\n");

        for (int i = 0; i < 4; i++)
        {
            printf("%.1f\t%.1f\t%.1f\t%.6f\n",
                   input_data[i * 2],
                   input_data[i * 2 + 1],
                   target_data[i],
                   output_data[i]);
        }

        //
        // ---------------- TRAINING ----------------
        //

        uint32_t batch_size = 4;
        uint16_t epochs = 100;
        uint16_t print_interval = 10;

        printf("\nTraining...\n");

        float loss;

        for (int i = 0; i < epochs; i++)
        {
            aialgo_train_model(
                &model,
                &input_tensor,
                &target_tensor,
                opt,
                batch_size);

            if (i % print_interval == 0)
            {
                aialgo_calc_loss_model_f32(
                    &model,
                    &input_tensor,
                    &target_tensor,
                    &loss);

                printf("Epoch %d Loss %f\n", i, loss);
            }
        }

        printf("\nTraining finished\n");

        //
        // ---------------- AFTER TRAIN ----------------
        //

        aialgo_inference_model(
            &model,
            &input_tensor,
            &output_tensor);

        printf("\nAfter training:\n");
        printf("x1\tx2\ty\ty_pred\n");

        for (int i = 0; i < 4; i++)
        {
            printf("%.1f\t%.1f\t%.1f\t%.6f\n",
                   input_data[i * 2],
                   input_data[i * 2 + 1],
                   target_data[i],
                   output_data[i]);
        }

        free(param_mem);
        free(mem);

        printf("\nType 'training' to run again\n");
    }

    return 0;
}