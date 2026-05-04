---
tags: [embedded, federated-learning, mqtt, gesture-recognition]
dataset: [custom-imu-data]
framework: [aifes, arduino]
---
# AIfES x Flower - Gesture Recognition with Federated Learning

This example demonstrates federated learning on embedded devices using [AIfES](https://www.aifes.ai) (Artificial Intelligence for Embedded Systems) and [Flower](https://flower.ai/). The application performs gesture recognition directly on the **M5StickC Plus 2** microcontroller using IMU sensor data and synchronizes the trained model with a federated learning server via MQTT.

## Project Overview

This demo trains a small neural network directly on the device, synchronizes the local model with a server using federated learning, and then classifies gestures in real-time. The system can recognize three different gestures after recording only 5 training samples per class.

### Key Features

- **On-device Training**: Train a neural network directly on the microcontroller with AIfES
- **Federated Learning**: Synchronize local models with a global model using Flower and MQTT
- **Automatic Gesture Detection**: IMU-based motion detection without button input
- **Real-time Classification**: Classify gestures on the device after federated synchronization

## Set up the project

### Prerequisites

**Hardware:**
- M5StickC Plus 2 microcontroller

**Software:**
- Arduino IDE (version 1.8.x or 2.x)
- Python 3.8+ (for the federated learning server)

### Fetch the app

Install Flower and required Python dependencies:

```shell
pip install flwr torch paho-mqtt numpy
```

You can obtain the example source code in two ways:

1. **Install the AIfES for Arduino library in the Arduino IDE**

   - Install the **AIfES for Arduino** library using the Arduino IDE. For details on setting up the Arduino IDE and AIfES for Arduino, see [Arduino Dependencies](#arduino-dependencies).
   - Locate the Arduino libraries folder as described in the Arduino documentation under “Manual installation”: <https://docs.arduino.cc/software/ide-v1/tutorials/installing-libraries/#manual-installation>.
   - In the libraries folder, navigate to:  
     `AIfES_for_Arduino/examples/5_M5_Stick_C_Plus_2_ESP32/0_Flower_AIfES_MQTT`

2. **Clone the GitHub repository**

   You can download the files for this example directly from GitHub and open the example folder:

   ```shell
   git clone https://github.com/Fraunhofer-IMS/AIfES_for_Arduino
   cd AIfES_for_Arduino/examples/5_M5_Stick_C_Plus_2_ESP32/0_Flower_AIfES_MQTT
   ```

### Project Structure

```
0_Flower_AIfES_MQTT
├── 0_Flower_AIfES_MQTT.ino      # Main firmware for M5StickC Plus 2
├── aifes_f32_fnn.h              # Neural network configuration
├── fl_comm_mqtt.h               # MQTT communication layer
├── fl_server_mqtt.py            # Federated learning server
├── pyproject.toml               # Python project dependencies (optional)
└── README.md
```

### Install dependencies

#### Arduino Dependencies

1. Install the **Arduino IDE** from [arduino.cc/en/software](https://www.arduino.cc/en/software)
2. Add the M5StickC Plus 2 board support via **Board Manager**
3. Install the following libraries via **Library Manager**:
   - [AIfES for Arduino](https://github.com/Fraunhofer-IMS/AIfES_for_Arduino) >= Version 2.2.1
   - `M5StickCPlus2` library
   - `PubSubClient` (for MQTT)

#### Configure WiFi and MQTT

Open `fl_comm_mqtt.h` and configure your WiFi credentials:

```c
#define FL_WIFI_SSID     "YourSSID"
#define FL_WIFI_PASSWORD "YourPassword"
```

Optionally, configure the MQTT broker (default: `broker.hivemq.com`):

```c
#define FL_MQTT_BROKER_HOST "broker.hivemq.com"
#define FL_MQTT_BROKER_PORT 1883
```

## Run the Federated Learning Server

Start the federated learning server on your computer:

```bash
python fl_server_mqtt.py
```

The server will:
- Connect to the MQTT broker
- Publish the initial global model
- Wait for client updates
- Aggregate models using FedAvg when at least 2 clients have sent updates
- Publish the updated global model

You should see output like:

```
Connecting to MQTT broker broker.hivemq.com:1883
Topic prefix: demo/fl
Connected to MQTT broker: reason_code=0
```

## Run on the M5StickC Plus 2

### Upload the Firmware

1. Open the example in the Arduino IDE via **File → Examples → AIfES for Arduino → 5_M5_Stick_C_Plus_2_ESP32 → 0_Flower_AIfES_MQTT**
2. Select **Tools → Board → M5StickC Plus 2**
3. Select the correct serial port under **Tools → Port**
4. Click **Upload** to flash the firmware
5. Optionally, if not done before, update the settings in the `fl_comm_mqtt.h` file.

### Record Training Data

After uploading, the device will guide you through data collection:

1. Hold the device in the **same position** throughout the process
2. Perform **Gesture 1** five times
   - Try to perform the gesture consistently
   - Wait briefly between repetitions
   - Ensure the motion exceeds the acceleration threshold
3. Perform **Gesture 2** five times
4. Perform **Gesture 3** five times

**What gestures work well?**
- Arm flex
- Quick punch motion
- Twist or rotation
- Sideways movement

### Training and Synchronization

After data collection, the device will:
1. Connect to WiFi
2. Download the initial global model from the MQTT broker
3. Train the local model with the recorded data
4. Upload the local model to the federated learning server
5. Wait for the aggregated global model
6. Switch to classification mode

### Classification Mode

Once in classification mode:
- Perform one of your trained gestures
- The display shows the predicted probabilities for all three classes
- Example output:
  ```
  Predicted gesture:
  Gesture 1: 85%
  Gesture 2: 10%
  Gesture 3: 05%
  ```

## Configuration Options

### Model Architecture

You can modify the neural network architecture in `aifes_f32_fnn.h`:

```c
#define DENSE_NEURONS_1 4    // Neurons in first hidden layer
#define DENSE_NEURONS_2 3    // Neurons in output layer (number of gestures)
```

### Data Collection Parameters

Adjust these constants in `01_Flower_AIfES_MQTT.ino`:

```c
#define NUM_GESTURES 3           // Number of gesture classes
#define NUM_SAMPLES 20           // IMU samples per gesture
#define SAMPLES_PER_CLASS 5      // Training samples per class
#define ACCEL_THRESHOLD 4        // Motion detection threshold
```

### Federated Learning Settings

Modify server settings in `fl_server_mqtt.py`:

```python
MIN_CLIENTS = 2              # Minimum clients for aggregation
MQTT_BROKER_HOST = "..."     # MQTT broker address
MQTT_TOPIC_PREFIX = "demo/fl" # Topic namespace
```

## Troubleshooting

**WiFi connection fails:**
- Check SSID and password in `fl_comm_mqtt.h`
- Ensure 2.4 GHz WiFi network (M5StickC Plus 2 doesn't support 5 GHz)

**Model download timeout:**
- Ensure the Python server is running
- Check MQTT broker connectivity
- Verify the topic prefix matches between client and server

**Gesture recording not starting:**
- Increase `ACCEL_THRESHOLD` if gestures are too subtle
- Decrease `ACCEL_THRESHOLD` if motion starts recording unintentionally

**Model aggregation not happening:**
- Ensure at least `MIN_CLIENTS` have sent updates
- Check server logs for error messages

## Advanced Features

### Calibration

To remove gravity offset, uncomment the calibration in `setup()`:

```c
calibrate();
```

### Secure MQTT Connection

For production use, configure TLS:

```c
#define FL_MQTT_BROKER_PORT 8883
// Configure WiFiClientSecure instead of WiFiClient
```

### Multiple Devices

To train with multiple M5StickC devices:
1. Flash the firmware to multiple devices
2. Ensure they all connect to the same MQTT broker
3. The server will aggregate models from all clients

## Resources

- **AIfES Website**: [aifes.ai](https://www.aifes.ai)
- **AIfES for Arduino**: [GitHub Repository](https://github.com/Fraunhofer-IMS/AIfES_for_Arduino)
- **Flower Framework**: [flower.ai](https://flower.ai/)
- **Flower Documentation**: [flower.ai/docs](https://flower.ai/docs/)
- **HiveMQ MQTT Broker**: [hivemq.com](https://www.hivemq.com)
- **M5StickC Plus 2**: [M5Stack Documentation](https://docs.m5stack.com/en/core/M5StickC%20PLUS2)

### Community

- **Flower Slack**: [Join Slack](https://flower.ai/join-slack/)
- **Flower Discuss**: [discuss.flower.ai](https://discuss.flower.ai/)
- **AIfES Tutorials**: [Arduino Project Hub](https://create.arduino.cc/projecthub/aifes_team)

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).

Copyright (C) 2020-2026 Fraunhofer Institute for Microelectronic Circuits and Systems.

See the license headers in the source files for more details.

## Citation

If you use this project in your research, please cite:

```bibtex
@ARTICLE{10403985,
  author={Wulfert, Lars and Kühnel, Johannes and Krupp, Lukas and Viga, Justus and Wiede, Christian and Gembaczka, Pierre and Grabmaier, Anton},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={AIfES: A Next-Generation Edge AI Framework}, 
  year={2024},
  volume={},
  number={},
  pages={1-16},
  keywords={Training;Data models;Artificial intelligence;Support vector machines;Hardware acceleration;Libraries;Performance evaluation;Machine Learning Framework;Edge AI Framework;On-Device Training;Embedded Systems;Resource-Constrained Devices;TinyML},
  doi={10.1109/TPAMI.2024.3355495}}
```


