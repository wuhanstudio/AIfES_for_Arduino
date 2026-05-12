/*
  AIfES MNIST CNN Training (Linux C Full Example)

  Input: 1x28x28 MNIST
  Model:
    Conv2D(4,3x3) -> ReLU -> MaxPool(2x2)
    Conv2D(8,3x3) -> ReLU -> MaxPool(2x2)
    Flatten -> Dense(10) -> Softmax
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "aifes.h"

#include "logging.h"

#define BATCH_SIZE 5
#define IMG_SIZE (28 * 28)
#define NUM_CLASSES 10

#define MNIST_TRAIN_IMG_COUNT 60000
#define MNIST_TEST_IMG_COUNT  10000

#define DISK_MOUNT_PT "./mnist"
#define MNIST_TRAIN_IMG_PATH DISK_MOUNT_PT"/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABEL_PATH DISK_MOUNT_PT"/train-labels-idx1-ubyte"

#define MNIST_TEST_IMG_PATH DISK_MOUNT_PT"/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABEL_PATH DISK_MOUNT_PT"/t10k-labels-idx1-ubyte"

// ------------------------------------------------------------
// MNIST LOADER
// ------------------------------------------------------------
int load_mnist_images(
    const char *filename,
    float *out,
    int count)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Cannot open %s\n", filename);
        return -1;
    }

    uint32_t magic, num, rows, cols;

    fread(&magic, 4, 1, fp);
    fread(&num, 4, 1, fp);
    fread(&rows, 4, 1, fp);
    fread(&cols, 4, 1, fp);

    magic = __builtin_bswap32(magic);
    num   = __builtin_bswap32(num);
    rows  = __builtin_bswap32(rows);
    cols  = __builtin_bswap32(cols);

    if (magic != 2051) {
        printf("Invalid MNIST image file\n");
        fclose(fp);
        return -1;
    }

    int size = rows * cols;
    for (int i = 0; i < count; i++) {
        if (i % (count / 10) == 0) {
            printf("Loading image %d/%d\n", i, count);
        }
        unsigned char buffer[784];
        fread(buffer, 1, size, fp);

        for (int j = 0; j < size; j++) {
            out[i * size + j] = buffer[j] / 255.0f;
        }
    }

    fclose(fp);
    return 0;
}

int load_mnist_labels(
    const char *filename,
    uint8_t *out,
    int count)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Cannot open %s\n", filename);
        return -1;
    }

    uint32_t magic, num;

    fread(&magic, 4, 1, fp);
    fread(&num, 4, 1, fp);

    magic = __builtin_bswap32(magic);
    num   = __builtin_bswap32(num);

    if (magic != 2049) {
        printf("Invalid MNIST label file\n");
        fclose(fp);
        return -1;
    }

    fread(out, 1, count, fp);

    fclose(fp);
    return 0;
}

// ASCII lib from (https://www.jianshu.com/p/1f58a0ebf5d9)
static const char codeLib[] = "@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.   ";
void mnist_print_img(float* buf)
{
    for(int y = 0; y < 28; y++) 
    {
        for (int x = 0; x < 28; x++) 
        {
            int index = 0; 
            if(buf[y*28+x] > (75 / 255.0f)) index =69;
            if(index < 0) index = 0;
                printf("%c",codeLib[index]);
                printf("%c",codeLib[index]);
        }
        printf("\n");
    }
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------

int main(void)
{
    printf("AIfES MNIST CNN Training\n\n");

    srand(time(NULL));

    // --------------------------------------------------------
    // DATA
    // --------------------------------------------------------

    float* input_data = (float*) malloc(MNIST_TRAIN_IMG_COUNT * IMG_SIZE * sizeof(float));
    uint8_t* labels = (uint8_t*) malloc(MNIST_TRAIN_IMG_COUNT * sizeof(uint8_t));
    
    load_mnist_images(
        "mnist/train-images-idx3-ubyte",
        input_data,
        MNIST_TRAIN_IMG_COUNT);
        
    load_mnist_labels(
        "mnist/train-labels-idx1-ubyte",
        labels,
        MNIST_TRAIN_IMG_COUNT);
    
    // One-hot targets
    float target_data[MNIST_TRAIN_IMG_COUNT * NUM_CLASSES];
    memset(target_data, 0, sizeof(target_data));
    
    for (int i = 0; i < MNIST_TRAIN_IMG_COUNT; i++) {
        target_data[i * NUM_CLASSES + labels[i]] = 1.0f;
    }

    // Print first 5 images and labels
    // for (int i = 0; i < 5; i++) {
    //     printf("\n--- Image %d ---\n", i);
    //     mnist_print_img(&input_data[i * IMG_SIZE]);
    //     printf("Label: %d\n", labels[i]);
    // }

    // --------------------------------------------------------
    // TENSORS
    // --------------------------------------------------------
    
    uint16_t input_shape[] = {MNIST_TRAIN_IMG_COUNT, 1, 28, 28};

    aitensor_t input_tensor =
        AITENSOR_4D_F32(input_shape, input_data);

    uint16_t target_shape[] = {MNIST_TRAIN_IMG_COUNT, NUM_CLASSES};

    aitensor_t target_tensor =
        AITENSOR_2D_F32(target_shape, target_data);

    // --------------------------------------------------------
    // LAYERS
    // --------------------------------------------------------

    uint16_t input_layer_shape[] = {1, 1, 28, 28};

    ailayer_input_f32_t input_layer =
        AILAYER_INPUT_F32_A(4, input_layer_shape);

    ailayer_conv2d_t conv1 =
        AILAYER_CONV2D_F32_A(
            4,
            HW(3,3),
            HW(1,1),
            HW(1,1),
            HW(0,0));

    ailayer_relu_f32_t relu1 =
        AILAYER_RELU_F32_A();

    ailayer_maxpool2d_t pool1 =
        AILAYER_MAXPOOL2D_F32_A(
            HW(2,2),
            HW(2,2),
            HW(0,0));

    ailayer_conv2d_t conv2 =
        AILAYER_CONV2D_F32_A(
            8,
            HW(3,3),
            HW(1,1),
            HW(1,1),
            HW(0,0));

    ailayer_relu_f32_t relu2 =
        AILAYER_RELU_F32_A();

    ailayer_maxpool2d_t pool2 =
        AILAYER_MAXPOOL2D_F32_A(
            HW(2,2),
            HW(2,2),
            HW(0,0));

    ailayer_flatten_t flatten =
        AILAYER_FLATTEN_F32_A();

    ailayer_dense_f32_t dense =
        AILAYER_DENSE_F32_A(NUM_CLASSES);

    ailayer_softmax_f32_t softmax =
        AILAYER_SOFTMAX_F32_A();

    ailoss_crossentropy_f32_t loss_fn;

    // --------------------------------------------------------
    // MODEL
    // --------------------------------------------------------

    aimodel_t model;
    ailayer_t *x;

    model.input_layer =
        ailayer_input_f32_default(&input_layer);

    x = ailayer_conv2d_chw_f32_default(&conv1, model.input_layer);
    x = ailayer_relu_f32_default(&relu1, x);
    x = ailayer_maxpool2d_chw_f32_default(&pool1, x);

    x = ailayer_conv2d_chw_f32_default(&conv2, x);
    x = ailayer_relu_f32_default(&relu2, x);
    x = ailayer_maxpool2d_chw_f32_default(&pool2, x);

    x = ailayer_flatten_f32_default(&flatten, x);
    x = ailayer_dense_f32_default(&dense, x);
    x = ailayer_softmax_f32_default(&softmax, x);

    model.output_layer = x;

    model.loss =
        ailoss_crossentropy_f32_default(
            &loss_fn,
            model.output_layer);

    aialgo_compile_model(&model);

    printf("\n--- Model ---\n");
    aialgo_print_model_structure(&model);

    // --------------------------------------------------------
    // MEMORY
    // --------------------------------------------------------

    uint32_t param_size =
        aialgo_sizeof_parameter_memory(&model);

    void *param_mem = malloc(param_size);

    aialgo_distribute_parameter_memory(
        &model,
        param_mem,
        param_size);

    aialgo_initialize_parameters_model(&model);

    // --------------------------------------------------------
    // OPTIMIZER
    // --------------------------------------------------------

    aiopti_adam_f32_t adam =
        AIOPTI_ADAM_F32(
            0.01f,
            0.9f,
            0.999f,
            1e-5f);

    aiopti_t *opt =
        aiopti_adam_f32_default(&adam);

    // --------------------------------------------------------
    // TRAIN MEMORY
    // --------------------------------------------------------

    uint32_t mem_size =
        aialgo_sizeof_training_memory(&model, opt);

    void *mem = malloc(mem_size);

    aialgo_schedule_training_memory(
        &model,
        opt,
        mem,
        mem_size);

    aialgo_init_model_for_training(
        &model,
        opt);

    // --------------------------------------------------------
    // TRAINING
    // --------------------------------------------------------

    printf("\nStart training...\n");

    float loss;

    for (int e = 0; e < 10; e++) {

        aialgo_train_model(
            &model,
            &input_tensor,
            &target_tensor,
            opt,
            BATCH_SIZE);

        aialgo_calc_loss_model_f32(
            &model,
            &input_tensor,
            &target_tensor,
            &loss);

        printf("Epoch %d loss = %f\n", e, loss);
    }

    // --------------------------------------------------------
    // INFERENCE
    // --------------------------------------------------------

    float *test_input_data = (float*)malloc(MNIST_TEST_IMG_COUNT * IMG_SIZE * sizeof(float));
    uint8_t *test_labels = (uint8_t*)malloc(MNIST_TEST_IMG_COUNT * sizeof(uint8_t));

    load_mnist_images(
        "mnist/t10k-images-idx3-ubyte",
        test_input_data,
        MNIST_TEST_IMG_COUNT);

    load_mnist_labels(
        "mnist/t10k-labels-idx1-ubyte",
        test_labels,
        MNIST_TEST_IMG_COUNT);

    float *output_data = (float*)malloc(1 * NUM_CLASSES * sizeof(float));
    uint16_t out_shape[] = {1, NUM_CLASSES};

    // Test on one random test image
    int i = rand() % MNIST_TEST_IMG_COUNT;
    aitensor_t test_tensor = AITENSOR_4D_F32(out_shape, &test_input_data[i * IMG_SIZE]);

    aitensor_t output_tensor =
        AITENSOR_2D_F32(out_shape, output_data);

    // Print one test image
    printf("\n--- Test Image %d ---\n", i);
    mnist_print_img(&test_input_data[i * IMG_SIZE]);

    aialgo_inference_model(
        &model,
        &test_tensor,
        &output_tensor);

    printf("\nOutput:\n");
    print_aitensor(&output_tensor);

    // Argmax to get predicted class
    printf("\nPredicted classes:\n");
    int predicted_class = 0;
    float max_prob = output_data[0];
    for (int j = 1; j < NUM_CLASSES; j++) {
        if (output_data[j] > max_prob) {
            max_prob = output_data[j];
            predicted_class = j;
        }
    }
    printf("Sample %d: Predicted class = %d, True class = %d\n",
            i, predicted_class, test_labels[i]);

    // Evaluate on all test images
    int correct = 0;
    for (int i = 0; i < MNIST_TEST_IMG_COUNT; i++)
    {
        test_tensor.data = &test_input_data[i * IMG_SIZE];

        aialgo_inference_model(
            &model,
            &test_tensor,
            &output_tensor);

        int predicted_class = 0;
        float max_prob = output_data[0];
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (output_data[j] > max_prob) {
                max_prob = output_data[j];
                predicted_class = j;
            }
        }

        if (predicted_class == test_labels[i]) {
            correct++;
        }
    }
    printf("\nTest Accuracy: %.2f%%\n", (correct / (float)MNIST_TEST_IMG_COUNT) * 100.0f);

    // --------------------------------------------------------
    // CLEANUP
    // --------------------------------------------------------

    free(mem);
    free(input_data);
    free(labels);
    free(test_input_data);
    free(test_labels);
    free(param_mem);

    return 0;
}
