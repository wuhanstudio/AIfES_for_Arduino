/*
  www.aifes.ai
  https://github.com/Fraunhofer-IMS/AIfES_for_Arduino

  You can find more AIfES tutorials here:
  https://create.arduino.cc/projecthub/aifes_team

  AIfES: Configuration file automatically generated from AIfES-Converter
  https://github.com/Fraunhofer-IMS/AIfES-Converter

  You first need to initialize the model once with:
  ---------------------------------------------------------------------------
    uint8_t error;
    error = aifes_f32_fnn_create_model();
    if (error == 1)
    {
        //do something for error handling
        //e.g. while(true)
    }
  ---------------------------------------------------------------------------
  Please check the error. It is 1 if something inside the initialization failed. Check the output

  Paste the following code into your project and insert your inputs/features:
  ---------------------------------------------------------------------------
    float input_data[60]; // AIfES input data
    float output_data[3]; // AIfES output data

    aifes_f32_fnn_inference((float*) input_data, (float*) output_data);
  ---------------------------------------------------------------------------


  Copyright (C) 2026  Fraunhofer Institute for Microelectronic Circuits and Systems.
  All rights reserved.
  AIfES is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.
  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/

#include <aifes.h>

#ifndef AIFES_F32_FNN
#define AIFES_F32_FNN

#define DATASETS 1
#define INPUTS  60
#define OUTPUTS 3

#define DENSE_NEURONS_1 4
#define DENSE_NEURONS_2 3

#define NUMBER_TRAINING_DATA  15 // Define for number of training data,
                             // Is not shared between files because of Arduino IDE

aimodel_t model;

// Setup the optimizer
aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(0.1f,  // learning rate
                                              0.9f,   // beta1
                                              0.999f, // beta2
                                              1e-8f); // eps

aiopti_t *optimizer;

// Setup the loss
ailoss_crossentropy_t cross_entropy_loss;

// Layer definition
uint16_t input_layer_shape[] = {DATASETS, INPUTS};

ailayer_input_f32_t		input_layer	= AILAYER_INPUT_F32_A(2, input_layer_shape);
ailayer_dense_f32_t		dense_layer_1	= AILAYER_DENSE_F32_A(DENSE_NEURONS_1);
ailayer_sigmoid_f32_t		sigmoid_layer_1	= AILAYER_SIGMOID_F32_A();
ailayer_dense_f32_t		dense_layer_2	= AILAYER_DENSE_F32_A(DENSE_NEURONS_2);
ailayer_softmax_f32_t		softmax_layer_2	= AILAYER_SOFTMAX_F32_A();


uint16_t output_layer_shape[] = {DATASETS, OUTPUTS};

uint8_t aifes_f32_fnn_create_model()
{
  ailayer_t *x;

  model.input_layer = ailayer_input_f32_default(&input_layer);
  x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
  x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
  x = ailayer_dense_f32_default(&dense_layer_2, x);
  x = ailayer_softmax_f32_default(&softmax_layer_2, x);
  model.output_layer = x;

  aialgo_compile_model(&model); // Compile the AIfES model

  // Setup of loss function
  model.loss = ailoss_crossentropy_f32_default(&cross_entropy_loss, model.output_layer);

  // Setup the optimizer
  optimizer = aiopti_adam_f32_default(&adam_opti);

  uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);

  void *parameter_memory = malloc(parameter_memory_size);

  if(parameter_memory == NULL)
  {
    AIPRINT("Not enough memory available for parameter memory of the neural network.");
    return 1;
  }

  aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

  // Set the seed for your configured random function for example with
  srand(42);
  aialgo_initialize_parameters_model(&model);

  uint32_t training_memory_size = aialgo_sizeof_training_memory(&model, optimizer);
  void *training_memory = malloc(training_memory_size);

  if(training_memory == NULL)
  {
    AIPRINT("Not enough memory available for parameter memory of the neural network.");
    return 1;
  }

  // Schedule the memory to the model
  aialgo_schedule_training_memory(&model, optimizer, training_memory, training_memory_size);

  // Initialize the training memory
  aialgo_init_model_for_training(&model, optimizer);
  return 0;

}

void aifes_f32_fnn_inference(float* input_data, float* output_data)
{

  aitensor_t input_tensor = AITENSOR_2D_F32(input_layer_shape, input_data);

  // Definition of the output shape
  aitensor_t output_tensor = AITENSOR_2D_F32(output_layer_shape, output_data);

  aialgo_inference_model(&model, &input_tensor, &output_tensor);

}

void aifes_f32_fnn_training(float* input_data, float* target_data)
{
  uint16_t input_tensor_shape[2] = {NUMBER_TRAINING_DATA, INPUTS};
  aitensor_t input_tensor = AITENSOR_2D_F32(input_tensor_shape, input_data);

  uint16_t target_tensor_shape[2] = {NUMBER_TRAINING_DATA, OUTPUTS};
  aitensor_t target_tensor = AITENSOR_2D_F32(target_tensor_shape, target_data);

  uint16_t output_tensor_shape[2] = {NUMBER_TRAINING_DATA, OUTPUTS};
  float output_data[6][2];
  aitensor_t output_tensor = AITENSOR_2D_F32(output_tensor_shape, output_data);

  int epochs = 1000;
  int batch_size = NUMBER_TRAINING_DATA;
  int print_interval = 10;
  float early_stopping_threshold = 0.09;

  float loss;
  AIPRINT("\nStart training\n");
  for(int i = 0; i < epochs; i++)
  {

    // One epoch of training. Iterates through the whole data once
    aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);
    
    aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);

    if(loss < early_stopping_threshold)
    {
      // Target loss reached. Training can be stopped
      break;
    }

    // Calculate and print loss every print_interval epochs
    if(i % print_interval == 0)
    {
      // Print the loss to the console
      AIPRINT("Epoch "); AIPRINT_INT("%5d", i);
      AIPRINT(": test loss: "); AIPRINT_FLOAT("%f", loss);
      AIPRINT("\n");
    }

  }
  AIPRINT("Finished training\n\n");
}

#endif // AIFES_F32_FNN