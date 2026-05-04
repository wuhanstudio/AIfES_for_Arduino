 /*
  www.aifes.ai
  https://github.com/Fraunhofer-IMS/AIfES_for_Arduino
  Copyright (C) 2020-2026  Fraunhofer Institute for Microelectronic Circuits and Systems.
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

  AIfES x Flower simple gesture recognition demo with federated learning
  --------------------

    Versions:
    1.0.0   Initial version

  What does the example:
  - This demo records and classifies three different simple gestures using the IMU of the M5StickC Plus 2
  - Gesture data is detected automatically via an acceleration threshold, so no button input is required
  - For each of the three gestures, 5 training samples are recorded
  - Each sample consists of 20 acceleration measurements on the x, y, and z axes
  - After data collection, the device connects to WiFi and downloads the initial global model
  - The recorded data is used to train an artificial neural network directly on the device with AIfES
  - The locally trained model is then synchronized with a federated learning Flower server via MQTT
  - After federated synchronization, the device switches automatically to classification mode
  - In classification mode, new gestures are detected automatically and classified on the device

  What gestures can be trained?
  - Simple gestures can be trained, such as an arm flex, a punch, a twist or a quick sideways movement

  Instructions:
  - Make sure that the board is always held in the same position in your hand
  - Follow the instructions shown on the display during data acquisition
  - Perform the first gesture 5 times
    * Try to perform the gesture as consistently as possible
    * Make a short pause between repetitions
    * Make sure that the movement exceeds the acceleration threshold so that recording starts
  - After that, perform the second gesture 5 times
  - Then perform the third gesture 5 times
  - Once all gesture data has been recorded, the neural network is created automatically
  - The device then connects to WiFi and downloads the initial global model
  - Local training starts automatically after a short countdown
  - After training, the local model is sent to the federated learning server and the aggregated global model is downloaded
  - Finally, the device enters classification mode automatically
  - In classification mode, perform one of the trained gestures to see the predicted probabilities for all three gesture classes on the display

  The example is based on a project from EloquentArduino:
  - https://eloquentarduino.github.io/2019/12/how-to-do-gesture-identification-on-arduino/
  - https://github.com/eloquentarduino/EloquentArduino/blob/master/examples/MicromlGestureIdentificationExample/MicromlGestureIdentificationExample.ino

  Calibration:
  - There is a simple calibration procedure to remove the fixed offset due to gravity
  - By default the calibration is commented out

  Hardware:
    M5 Stick C Plus 2
    
  You can find more AIfES tutorials here:
  https://create.arduino.cc/projecthub/aifes_team
  */

#include "M5StickCPlus2.h"
#include "aifes_f32_fnn.h"
#include "fl_comm_mqtt.h"

// Number of gesture classes to train and classify
#define NUM_GESTURES 3

// Number of IMU samples recorded per gesture instance
#define NUM_SAMPLES 20  //15 30

// Number of acceleration axes used: x, y, z
#define NUM_AXES 3

// Total number of input features per recorded gesture
#define NUM_FEATURES NUM_SAMPLES * NUM_AXES

// Maximum absolute value used to clamp acceleration values
#define TRUNCATE 20

// Motion threshold used to detect the start of a gesture
#define ACCEL_THRESHOLD 4 //5

// Delay in milliseconds between two IMU samples
#define INTERVAL 30

// Number of training examples recorded per gesture class
#define SAMPLES_PER_CLASS 5

// Total number of recorded training samples
#define NUMBER_OF_DATA  NUM_GESTURES * SAMPLES_PER_CLASS

// One-hot encoded label of the currently recorded gesture class
// Gesture 1 -> {1, 0, 0}
// Gesture 2 -> {0, 1, 0}
// Gesture 3 -> {0, 0, 1}
int labels_print[3] = {1,0,0};
//int labels_print[2] = {1,0};

// Training input data
// Stores all recorded gesture feature vectors
float training_data[NUMBER_OF_DATA][NUM_FEATURES];

// Training target data
// Stores the one-hot encoded labels for each recorded gesture
float labels[NUMBER_OF_DATA][NUM_GESTURES];

// Temporary feature buffer for the currently recorded gesture
float features[NUM_FEATURES];

// Counts how many training samples have already been stored
int training_data_counter = 0;

// Counts the currently recorded gesture class
int class_counter = 0;

// Counts how many samples have been recorded for the current class
int sample_counter = 0;

// Baseline values used for optional gravity calibration
float baseline[NUM_AXES];

// Current federated learning round number
uint32_t g_fl_round = 0;


/**
 * Initializes the device, records gesture data, trains the local model,
 * synchronizes the trained weights with the federated learning server,
 * and prepares the board for gesture classification.
 */
void setup() {
  Serial.begin(115200);

  // Initialize the M5StickC Plus 2 device
  auto cfg = M5.config();
  StickCP2.begin(cfg);

  // Configure display settings
  StickCP2.Display.setRotation(3);
  StickCP2.Display.setTextColor(GREEN);
  StickCP2.Display.setTextDatum(middle_center);
  StickCP2.Display.setFont(&fonts::FreeSansBold9pt7b);
  StickCP2.Display.setTextSize(1);

  // Print instructions on the display
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.printf("AIfES x Flower\n");
  StickCP2.Display.printf("Hold the board in the same position in your hand!\n");

  // AIfES requires randomized initial weights for training
  // A fixed seed is used here to ensure reproducibility
  srand(42);

  // Optional gravity calibration
  // calibrate();
  
  float ax, ay, az; 

  // Prompt the user to record the first gesture
  StickCP2.Display.setCursor(0, 80);
  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.printf("Repeat the first gesture five times!\n");

  bool motion_detected = false;

  // Record gesture data for all gesture classes
  while(class_counter <= (NUM_GESTURES - 1)) {

    // Wait until motion above the threshold is detected
    while(!motion_detected) {   
      auto imu_update = StickCP2.Imu.update();
      if (imu_update) {     
        auto data = StickCP2.Imu.getImuData();
        ax = constrain(data.accel.x - baseline[0], -TRUNCATE, TRUNCATE);
        ay = constrain(data.accel.y - baseline[1], -TRUNCATE, TRUNCATE);
        az = constrain(data.accel.z - baseline[2], -TRUNCATE, TRUNCATE);
      }
        
      if (motionDetected(ax, ay, az)) {
        motion_detected = true;
      } else {
        delay(10);
      }
    }

    // Record the detected gesture and store its features
    if(class_counter <= (NUM_GESTURES - 1)) {
      recordIMU();
      SafeFeatures();
      motion_detected = false;
    }

    // Update counters after one sample has been recorded
    sample_counter = sample_counter + 1;
    training_data_counter = training_data_counter + 1;
    
    // After enough samples for one class, switch to the next class
    if(sample_counter == SAMPLES_PER_CLASS) {
      sample_counter = 0;
      class_counter = class_counter + 1;
      
      if(class_counter == 1) {
        StickCP2.Display.clear();
        StickCP2.Display.setCursor(0, 10);
        StickCP2.Display.setTextColor(GREEN);
        StickCP2.Display.printf("AIfES x Flower\n");
        StickCP2.Display.printf("Hold the board in the same position in your hand!\n");
        StickCP2.Display.setTextColor(BLUE);
        StickCP2.Display.setCursor(0, 80);
        StickCP2.Display.printf("Repeat the second gesture five times!");

        // Set one-hot label for gesture 2
        labels_print[0] = 0;
        labels_print[1] = 1;
        labels_print[2] = 0;
      }
      
      if(class_counter == 2) {
        StickCP2.Display.clear();
        StickCP2.Display.setCursor(0, 10);
        StickCP2.Display.setTextColor(GREEN);
        StickCP2.Display.printf("AIfES x Flower\n");
        StickCP2.Display.printf("Hold the board in the same position in your hand!\n");
        StickCP2.Display.setTextColor(BLUE);
        StickCP2.Display.setCursor(0, 80);
        StickCP2.Display.printf("Repeat the third gesture five times!");

        // Set one-hot label for gesture 3
        labels_print[0] = 0;
        labels_print[1] = 0;
        labels_print[2] = 1;
      }
      
      if(class_counter > 2) {
        StickCP2.Display.clear();
        StickCP2.Display.setTextColor(GREEN);
        StickCP2.Display.setCursor(0, 10);
        StickCP2.Display.printf("AIfES x Flower\n");
        StickCP2.Display.setTextColor(BLUE);
        StickCP2.Display.printf("Data recording finished!");
      }
    }

    delay(50);
  }

  // Create the AIfES neural network model
  uint8_t error;
  error = aifes_f32_fnn_create_model();
  if (error == 1) {
    // Stop execution if model initialization fails
    StickCP2.Display.setTextColor(RED);
    StickCP2.Display.printf("Init of network failed!");
    while(true);
  }

  // Ask the user to connect to WiFi
  StickCP2.Display.clear();
  StickCP2.Display.setTextColor(GREEN);
  StickCP2.Display.setTextSize(1);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.printf("AIfES x Flower\n");
  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.printf("Connect to WiFi!");

  bool error_wifi = false;
  error_wifi = fl_wifi_connect();
  if(error_wifi == false) {
    // Stop execution if WiFi initialization fails
    StickCP2.Display.setTextColor(RED);
    StickCP2.Display.printf("Init of WiFi failed!");
    while(true);
  }

  // Download the initial global model from the federated learning server
  StickCP2.Display.clear();
  StickCP2.Display.setTextColor(GREEN);
  StickCP2.Display.setTextSize(1);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.printf("AIfES x Flower\n");
  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.printf("Pull init FL model!");

  bool error_fl = false;
  error_fl = fl_download_initial_global_weights(&g_fl_round);
  if(error_fl == false) {
    // Stop execution if downloading the global model fails
    StickCP2.Display.setTextColor(RED);
    StickCP2.Display.printf("Download of global model failed!");
    while(true);
  }

  // Inform the user that data recording has finished
  StickCP2.Display.clear();
  StickCP2.Display.setTextColor(GREEN);
  StickCP2.Display.setTextSize(1);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.printf("AIfES x Flower\n");
  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.printf("Data recording finished!");
  delay(1000);

  // Display a short countdown before local training starts
  StickCP2.Display.setTextColor(BLUE);
  for(uint8_t i = 3; i > 0; i--) {
    StickCP2.Display.setCursor(0, 60);
    StickCP2.Display.setTextColor(BLACK);
    StickCP2.Display.printf("Start training in: %i seconds\n", i + 1);

    StickCP2.Display.setCursor(0, 60);
    StickCP2.Display.setTextColor(BLUE);
    StickCP2.Display.printf("Start training in: %i seconds\n", i);
    delay(1000);
  }
  
  StickCP2.Display.setCursor(0, 60);
  StickCP2.Display.setTextColor(BLACK);
  StickCP2.Display.printf("Start training in: %i seconds\n", 1);

  StickCP2.Display.setCursor(0, 60);
  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.printf("Training");

  // Train the local neural network with the recorded gesture data
  aifes_f32_fnn_training((float*)training_data, (float*) labels);

  // Upload the local model and wait for the aggregated global model
  StickCP2.Display.clear();
  StickCP2.Display.setTextColor(GREEN);
  StickCP2.Display.setTextSize(1);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.printf("AIfES x Flower\n");
  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.printf("Send local model! ");
  StickCP2.Display.printf("Pull global model after aggregation!");

  uint32_t new_round = g_fl_round;
  bool sync_ok = fl_federated_sync(g_fl_round, NUMBER_OF_DATA, &new_round);
  if (sync_ok) {
    g_fl_round = new_round;
    Serial.print("Federated sync successful, new round=");
    Serial.println(g_fl_round);
  } else {
    Serial.println("Federated sync failed or timed out");
  }

  // Switch to classification mode
  StickCP2.Display.clear();
  StickCP2.Display.setTextColor(GREEN);
  StickCP2.Display.setTextSize(1);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.printf("AIfES x Flower\n");
  StickCP2.Display.setCursor(0, 30);
  StickCP2.Display.setCursor(0, 80);
  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.printf("Perform your gesture.\n");
}


/**
 * Main runtime loop.
 *
 * Continuously reads acceleration values from the IMU, checks whether
 * motion above the threshold is detected, records the current gesture,
 * and classifies it using the trained neural network.
 */
void loop() {
  float ax, ay, az;

  // Update IMU values
  auto imu_update = StickCP2.Imu.update();
  if (imu_update) {     
    auto data = StickCP2.Imu.getImuData();
    ax = data.accel.x;
    ay = data.accel.y;
    az = data.accel.z;
  }

  // Remove baseline offset and clamp values
  ax = constrain(ax - baseline[0], -TRUNCATE, TRUNCATE);
  ay = constrain(ay - baseline[1], -TRUNCATE, TRUNCATE);
  az = constrain(az - baseline[2], -TRUNCATE, TRUNCATE);

  // Wait until motion is detected
  if (!motionDetected(ax, ay, az)) {
    delay(10);
    return;
  }
  
  // Record IMU data and classify the gesture
  recordIMU();
  classification();

  delay(500);
}


/**
 * Runs inference on the currently recorded gesture and displays
 * the predicted probabilities for all gesture classes.
 */
void classification() {
  float output[NUM_GESTURES];

  // Run neural network inference
  aifes_f32_fnn_inference(features, output);

  // Display classification result
  StickCP2.Display.clear();
  StickCP2.Display.setTextColor(GREEN);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.printf("AIfES x Flower\n");

  StickCP2.Display.setCursor(0, 30);
  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.printf("Predicted gesture:\n");
  StickCP2.Display.printf("Gesture 1: %02.0f%%\n", output[0] * 100.0f);
  StickCP2.Display.printf("Gesture 2: %02.0f%%\n", output[1] * 100.0f);
  StickCP2.Display.printf("Gesture 3: %02.0f%%\n", output[2] * 100.0f);

  StickCP2.Display.setTextColor(GREEN);
  StickCP2.Display.printf("Perform your next gesture.\n");
}


/**
 * Records one gesture sample from the IMU.
 *
 * For each sample step, the acceleration values of all three axes
 * are read, baseline-corrected, truncated, and stored in the
 * temporary feature buffer.
 */
void recordIMU() {
  float ax, ay, az;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    auto imu_update = StickCP2.Imu.update();
    if (imu_update) {     
      auto data = StickCP2.Imu.getImuData();

      ax = data.accel.x;
      ay = data.accel.y;
      az = data.accel.z;
    }
        
    // Remove baseline offset and clamp values
    ax = constrain(ax - baseline[0], -TRUNCATE, TRUNCATE);
    ay = constrain(ay - baseline[1], -TRUNCATE, TRUNCATE);
    az = constrain(az - baseline[2], -TRUNCATE, TRUNCATE);

    // Store x, y, and z values in the feature buffer
    features[i * NUM_AXES + 0] = ax;
    features[i * NUM_AXES + 1] = ay;
    features[i * NUM_AXES + 2] = az;

    delay(INTERVAL);
  }
}


/**
 * Checks whether the current acceleration values indicate motion.
 *
 * :param ax: Acceleration along the x-axis
 * :param ay: Acceleration along the y-axis
 * :param az: Acceleration along the z-axis
 * :return: True if the summed absolute acceleration exceeds the threshold
 */
bool motionDetected(float ax, float ay, float az) {
  return (abs(ax) + abs(ay) + abs(az)) > ACCEL_THRESHOLD;
}


/**
 * Stores the currently recorded feature vector in the training dataset
 * and assigns the corresponding one-hot encoded label.
 *
 * Also updates the display to show how many samples of the current
 * gesture class have already been recorded.
 */
void SafeFeatures() {
  for (int i = 0; i < NUM_FEATURES; i++) {
    training_data[training_data_counter][i] = features[i];
        
    if(i == NUM_FEATURES - 1) {
      labels[training_data_counter][0] = labels_print[0];
      labels[training_data_counter][1] = labels_print[1];
      labels[training_data_counter][2] = labels_print[2];
    }
  }

  // Update the display with the number of recorded samples
  StickCP2.Display.setCursor(0, 120);
  StickCP2.Display.setTextColor(BLACK);
  StickCP2.Display.printf("%i sample recorded!", (training_data_counter % 5));

  StickCP2.Display.setTextColor(BLUE);
  StickCP2.Display.setCursor(0, 120);
  StickCP2.Display.printf("%i sample recorded!", (training_data_counter % 5) + 1);
}