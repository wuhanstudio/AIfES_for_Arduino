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
*/
#pragma once

#include <WiFi.h>
#include <PubSubClient.h>
#include <stdint.h>
#include <string.h>
#include "aifes_f32_fnn.h"
#include "esp_mac.h"

// Default WiFi SSID used for the federated learning device connection
#ifndef FL_WIFI_SSID
#define FL_WIFI_SSID     "YourSSID"
#endif

// Default WiFi password used for the federated learning device connection
#ifndef FL_WIFI_PASSWORD
#define FL_WIFI_PASSWORD "YourPassword"
#endif

// Default MQTT broker host
#ifndef FL_MQTT_BROKER_HOST
#define FL_MQTT_BROKER_HOST "broker.hivemq.com"
#endif

// Default MQTT broker port
#ifndef FL_MQTT_BROKER_PORT
#define FL_MQTT_BROKER_PORT 1883
#endif

// Optional MQTT username
#ifndef FL_MQTT_USERNAME
#define FL_MQTT_USERNAME ""
#endif

// Optional MQTT password
#ifndef FL_MQTT_PASSWORD
#define FL_MQTT_PASSWORD ""
#endif

// Base MQTT topic prefix used for all federated learning communication
#ifndef FL_MQTT_TOPIC_PREFIX
#define FL_MQTT_TOPIC_PREFIX "demo/fl"
#endif

// MQTT keepalive interval in seconds
#ifndef FL_MQTT_KEEPALIVE_S
#define FL_MQTT_KEEPALIVE_S 60
#endif

// MQTT socket timeout in seconds
#ifndef FL_MQTT_SOCKET_TIMEOUT_S
#define FL_MQTT_SOCKET_TIMEOUT_S 15
#endif

// MQTT client buffer size in bytes
#ifndef FL_MQTT_BUFFER_SIZE
#define FL_MQTT_BUFFER_SIZE 1600
#endif

// Maximum waiting time for MQTT responses in milliseconds
#ifndef FL_MQTT_WAIT_TIMEOUT_MS
#define FL_MQTT_WAIT_TIMEOUT_MS 45000
#endif

// Number of neurons in the first dense layer
#define FL_DENSE1_UNITS 4

// Number of neurons in the second dense layer
#define FL_DENSE2_UNITS 3

// Number of weights in the first dense layer
#define FL_DENSE1_WEIGHTS_COUNT (INPUTS * FL_DENSE1_UNITS)

// Number of bias values in the first dense layer
#define FL_DENSE1_BIAS_COUNT    (FL_DENSE1_UNITS)

// Number of weights in the second dense layer
#define FL_DENSE2_WEIGHTS_COUNT (FL_DENSE1_UNITS * FL_DENSE2_UNITS)

// Number of bias values in the second dense layer
#define FL_DENSE2_BIAS_COUNT    (FL_DENSE2_UNITS)

// Total number of trainable parameters in the model
#define FL_TOTAL_WEIGHTS_COUNT  (FL_DENSE1_WEIGHTS_COUNT + FL_DENSE1_BIAS_COUNT + FL_DENSE2_WEIGHTS_COUNT + FL_DENSE2_BIAS_COUNT)


/**
 * Header of a global model packet received from the server.
 *
 * The packet contains the global training round number and
 * the total number of model weights included in the payload.
 */
struct FLGlobalPacketHeader {
  uint32_t round;
  uint32_t count;
};


/**
 * Header of a local model update packet sent to the server.
 *
 * The packet contains the local training round number, the
 * number of weights, and the number of local training examples
 * used on the device.
 */
struct FLUpdatePacketHeader {
  uint32_t round;
  uint32_t count;
  uint32_t num_examples;
};


// WiFi client used as the transport layer for MQTT communication
static WiFiClient fl_wifi_client;

// MQTT client used for federated learning communication
static PubSubClient fl_mqtt_client(fl_wifi_client);

// Indicates whether a valid global model packet has been received
static bool fl_has_global_packet = false;

// Stores the latest global training round received from the server
static uint32_t fl_latest_round = 0;

// Buffer containing the latest received global model weights
static float fl_latest_weights[FL_TOTAL_WEIGHTS_COUNT];

// MQTT topic for the latest global model
static char fl_topic_global[96];

// MQTT topic for sending local client updates
static char fl_topic_update[96];

// MQTT topic for receiving acknowledgments from the server
static char fl_topic_ack[96];

// MQTT topic for receiving server health messages
static char fl_topic_health[96];


// Buffer for the generated client identifier
static char client_id_buffer[32];

// Default fallback client identifier
const char* FL_CLIENT_ID = "esp32-default";


/**
 * Initializes the MQTT client identifier based on the device MAC address.
 *
 * The generated identifier uses the format:
 * esp32-XXXXXX
 * where the last three MAC bytes are appended in hexadecimal form.
 */
void init_client_id(void)
{
    uint8_t mac[6];
    esp_read_mac(mac, ESP_MAC_WIFI_STA);

    snprintf(client_id_buffer, sizeof(client_id_buffer),
             "esp32-%02X%02X%02X",
             mac[3], mac[4], mac[5]);

    FL_CLIENT_ID = client_id_buffer;
}


/**
 * Builds all MQTT topic strings used by the client.
 *
 * The topics are generated dynamically based on the configured
 * topic prefix and the unique client identifier.
 */
static inline void fl_build_topics() {
  snprintf(fl_topic_global, sizeof(fl_topic_global), "%s/global/latest", FL_MQTT_TOPIC_PREFIX);
  snprintf(fl_topic_update, sizeof(fl_topic_update), "%s/clients/%s/update", FL_MQTT_TOPIC_PREFIX, FL_CLIENT_ID);
  snprintf(fl_topic_ack, sizeof(fl_topic_ack), "%s/clients/%s/ack", FL_MQTT_TOPIC_PREFIX, FL_CLIENT_ID);
  snprintf(fl_topic_health, sizeof(fl_topic_health), "%s/server/health", FL_MQTT_TOPIC_PREFIX);
}


/**
 * Connects the device to the configured WiFi network.
 *
 * If the device is already connected, the function returns immediately.
 * Otherwise, it switches to station mode, starts the connection process,
 * and waits until either the connection succeeds or the timeout expires.
 *
 * :param timeout_ms: Maximum time to wait for the WiFi connection
 * :return: True if the WiFi connection was established successfully
 */
static inline bool fl_wifi_connect(uint32_t timeout_ms = 20000) {
  init_client_id();
  
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(FL_WIFI_SSID, FL_WIFI_PASSWORD);

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    if ((millis() - start) > timeout_ms) {
      return false;
    }
  }
  return true;
}


/**
 * MQTT callback function for incoming messages.
 *
 * This function handles three types of MQTT messages:
 * - global model packets
 * - server acknowledgment messages
 * - server health messages
 *
 * If a valid global model packet is received, the weights are copied
 * into the local buffer and marked as available.
 *
 * :param topic: MQTT topic of the received message
 * :param payload: Raw payload data
 * :param length: Length of the payload in bytes
 */
static inline void fl_mqtt_callback(char *topic, byte *payload, unsigned int length) {
  if (strcmp(topic, fl_topic_global) == 0) {
    if (length < sizeof(FLGlobalPacketHeader)) {
      Serial.println("Global packet too small");
      return;
    }

    FLGlobalPacketHeader hdr;
    memcpy(&hdr, payload, sizeof(hdr));

    if (hdr.count != FL_TOTAL_WEIGHTS_COUNT) {
      Serial.println("Global packet count mismatch");
      return;
    }

    const unsigned int expected = sizeof(FLGlobalPacketHeader) + hdr.count * sizeof(float);
    if (length != expected) {
      Serial.println("Global packet size mismatch");
      return;
    }

    memcpy(fl_latest_weights, payload + sizeof(FLGlobalPacketHeader), hdr.count * sizeof(float));
    fl_latest_round = hdr.round;
    fl_has_global_packet = true;

    Serial.print("Received global round via MQTT: ");
    Serial.println(fl_latest_round);
    return;
  }

  if (strcmp(topic, fl_topic_ack) == 0) {
    Serial.print("MQTT ACK: ");
    for (unsigned int i = 0; i < length; ++i) {
      Serial.print((char)payload[i]);
    }
    Serial.println();
    return;
  }

  if (strcmp(topic, fl_topic_health) == 0) {
    Serial.print("MQTT HEALTH: ");
    for (unsigned int i = 0; i < length; ++i) {
      Serial.print((char)payload[i]);
    }
    Serial.println();
  }
}


/**
 * Establishes the MQTT connection and subscribes to all required topics.
 *
 * The function ensures that WiFi is connected, initializes the topic strings,
 * configures the MQTT client, and subscribes to the global model, acknowledgment,
 * and health topics.
 *
 * :return: True if the MQTT connection and subscriptions were successful
 */
static inline bool fl_mqtt_connect() {
  if (!fl_wifi_connect()) {
    return false;
  }

  if (fl_topic_global[0] == '\0') {
    fl_build_topics();
  }

  fl_mqtt_client.setServer(FL_MQTT_BROKER_HOST, FL_MQTT_BROKER_PORT);
  fl_mqtt_client.setCallback(fl_mqtt_callback);
  fl_mqtt_client.setKeepAlive(FL_MQTT_KEEPALIVE_S);
  fl_mqtt_client.setSocketTimeout(FL_MQTT_SOCKET_TIMEOUT_S);
  fl_mqtt_client.setBufferSize(FL_MQTT_BUFFER_SIZE);

  if (fl_mqtt_client.connected()) {
    return true;
  }

  bool ok = false;
  if (strlen(FL_MQTT_USERNAME) > 0) {
    ok = fl_mqtt_client.connect(FL_CLIENT_ID, FL_MQTT_USERNAME, FL_MQTT_PASSWORD);
  } else {
    ok = fl_mqtt_client.connect(FL_CLIENT_ID);
  }

  if (!ok) {
    Serial.print("MQTT connect failed, rc=");
    Serial.println(fl_mqtt_client.state());
    return false;
  }

  bool sub1 = fl_mqtt_client.subscribe(fl_topic_global, 1);
  bool sub2 = fl_mqtt_client.subscribe(fl_topic_ack, 1);
  bool sub3 = fl_mqtt_client.subscribe(fl_topic_health, 1);

  Serial.print("Subscribed global=");
  Serial.print(sub1);
  Serial.print(" ack=");
  Serial.print(sub2);
  Serial.print(" health=");
  Serial.println(sub3);

  return sub1 && sub2 && sub3;
}


/**
 * Processes MQTT traffic for a specified amount of time.
 *
 * During the loop, the function keeps the MQTT client connected,
 * reconnects if necessary, and continuously processes incoming messages.
 *
 * :param timeout_ms: Duration in milliseconds for which MQTT messages should be processed
 * :return: Always true after the loop finishes
 */
static inline bool fl_mqtt_loop_for(uint32_t timeout_ms) {
  uint32_t start = millis();
  while ((millis() - start) < timeout_ms) {
    if (!fl_mqtt_client.connected()) {
      if (!fl_mqtt_connect()) {
        delay(250);
        continue;
      }
    }
    fl_mqtt_client.loop();
    delay(10);
  }
  return true;
}


/**
 * Returns the total number of model weights used in the network.
 *
 * :return: Total number of trainable parameters
 */
static inline size_t fl_get_weight_count() {
  return FL_TOTAL_WEIGHTS_COUNT;
}


/**
 * Exports the current local model weights into a flat output buffer.
 *
 * The weights are copied in the following order:
 * - dense layer 1 weights
 * - dense layer 1 bias
 * - dense layer 2 weights
 * - dense layer 2 bias
 *
 * :param out_weights: Target buffer for the exported weights
 * :param out_count: Size of the target buffer
 * :return: True if the export was successful
 */
static inline bool fl_export_model_weights(float *out_weights, size_t out_count) {
  if (out_weights == nullptr || out_count < FL_TOTAL_WEIGHTS_COUNT) {
    return false;
  }
  
  float *d1w = (float *)dense_layer_1.weights.data;
  float *d1b = (float *)dense_layer_1.bias.data;
  float *d2w = (float *)dense_layer_2.weights.data;
  float *d2b = (float *)dense_layer_2.bias.data;

  if (d1w == nullptr || d1b == nullptr || d2w == nullptr || d2b == nullptr) {
    return false;
  }

  size_t offset = 0;
  memcpy(out_weights + offset, d1w, FL_DENSE1_WEIGHTS_COUNT * sizeof(float));
  offset += FL_DENSE1_WEIGHTS_COUNT;
  memcpy(out_weights + offset, d1b, FL_DENSE1_BIAS_COUNT * sizeof(float));
  offset += FL_DENSE1_BIAS_COUNT;
  memcpy(out_weights + offset, d2w, FL_DENSE2_WEIGHTS_COUNT * sizeof(float));
  offset += FL_DENSE2_WEIGHTS_COUNT;
  memcpy(out_weights + offset, d2b, FL_DENSE2_BIAS_COUNT * sizeof(float));
  return true;
}


/**
 * Imports model weights from a flat input buffer into the local network.
 *
 * The weights are copied into the neural network in the following order:
 * - dense layer 1 weights
 * - dense layer 1 bias
 * - dense layer 2 weights
 * - dense layer 2 bias
 *
 * :param in_weights: Input buffer containing the model weights
 * :param in_count: Number of weights in the input buffer
 * :return: True if the import was successful
 */
static inline bool fl_import_model_weights(const float *in_weights, size_t in_count) {
  if (in_weights == nullptr || in_count != FL_TOTAL_WEIGHTS_COUNT) {
    return false;
  }

  float *d1w = (float *)dense_layer_1.weights.data;
  float *d1b = (float *)dense_layer_1.bias.data;
  float *d2w = (float *)dense_layer_2.weights.data;
  float *d2b = (float *)dense_layer_2.bias.data;

  if (d1w == nullptr || d1b == nullptr || d2w == nullptr || d2b == nullptr) {
    return false;
  }

  size_t offset = 0;
  memcpy(d1w, in_weights + offset, FL_DENSE1_WEIGHTS_COUNT * sizeof(float));
  offset += FL_DENSE1_WEIGHTS_COUNT;
  memcpy(d1b, in_weights + offset, FL_DENSE1_BIAS_COUNT * sizeof(float));
  offset += FL_DENSE1_BIAS_COUNT;
  memcpy(d2w, in_weights + offset, FL_DENSE2_WEIGHTS_COUNT * sizeof(float));
  offset += FL_DENSE2_WEIGHTS_COUNT;
  memcpy(d2b, in_weights + offset, FL_DENSE2_BIAS_COUNT * sizeof(float));
  return true;
}


/**
 * Downloads the initial global model weights from the MQTT broker.
 *
 * The function waits for a retained global model packet, verifies it,
 * imports the received weights into the local neural network,
 * and optionally returns the received server round number.
 *
 * :param server_round: Optional pointer to store the received global round
 * :param timeout_ms: Maximum waiting time for the retained model packet
 * :return: True if the download and import were successful
 */
static inline bool fl_download_initial_global_weights(uint32_t *server_round = nullptr, uint32_t timeout_ms = FL_MQTT_WAIT_TIMEOUT_MS) {
  if (!fl_mqtt_connect()) {
    return false;
  }

  fl_has_global_packet = false;
  if (!fl_mqtt_loop_for(timeout_ms)) {
    return false;
  }

  if (!fl_has_global_packet) {
    Serial.println("No retained global model received from broker");
    return false;
  }

  if (!fl_import_model_weights(fl_latest_weights, FL_TOTAL_WEIGHTS_COUNT)) {
    return false;
  }

  if (server_round != nullptr) {
    *server_round = fl_latest_round;
  }
  return true;
}


/**
 * Publishes the locally trained model weights to the server via MQTT.
 *
 * A binary update packet is created containing the current round,
 * the number of weights, the number of local training examples,
 * and the weight values themselves.
 *
 * :param round: Current training round of the client
 * :param weights: Local model weights to be uploaded
 * :param count: Number of model weights
 * :param num_examples: Number of local training examples
 * :return: True if the update was published successfully
 */
static inline bool fl_mqtt_post_local_weights(uint32_t round, const float *weights, size_t count, uint32_t num_examples) {
  if (!fl_mqtt_connect()) {
    return false;
  }

  const size_t payload_size = sizeof(FLUpdatePacketHeader) + count * sizeof(float);
  uint8_t *payload = (uint8_t *)malloc(payload_size);
  if (payload == nullptr) {
    return false;
  }

  FLUpdatePacketHeader hdr;
  hdr.round = round;
  hdr.count = (uint32_t)count;
  hdr.num_examples = num_examples;

  memcpy(payload, &hdr, sizeof(hdr));
  memcpy(payload + sizeof(hdr), weights, count * sizeof(float));

  bool ok = fl_mqtt_client.publish(fl_topic_update, payload, payload_size, false);
  free(payload);

  Serial.print("MQTT publish update ok=");
  Serial.println(ok);
  return ok;
}


/**
 * Performs one complete federated learning synchronization step.
 *
 * The function exports the current local model weights, uploads them
 * to the MQTT server, waits for a newer global model, and imports
 * the received global weights into the local model.
 *
 * :param current_round: Current local training round
 * :param num_examples: Number of local training examples used for the update
 * :param new_round: Optional pointer to store the new global round number
 * :param wait_timeout_ms: Maximum waiting time for the updated global model
 * :return: True if synchronization completed successfully
 */
static inline bool fl_federated_sync(uint32_t current_round, uint32_t num_examples, uint32_t *new_round = nullptr, uint32_t wait_timeout_ms = FL_MQTT_WAIT_TIMEOUT_MS) {
  float local_weights[FL_TOTAL_WEIGHTS_COUNT];

  if (!fl_export_model_weights(local_weights, FL_TOTAL_WEIGHTS_COUNT)) {
    return false;
  }

  if (!fl_mqtt_post_local_weights(current_round, local_weights, FL_TOTAL_WEIGHTS_COUNT, num_examples)) {
    return false;
  }

  uint32_t start = millis();
  while ((millis() - start) < wait_timeout_ms) {
    if (!fl_mqtt_client.connected()) {
      if (!fl_mqtt_connect()) {
        delay(250);
        continue;
      }
    }
    fl_mqtt_client.loop();
    if (fl_has_global_packet && fl_latest_round > current_round) {
      if (!fl_import_model_weights(fl_latest_weights, FL_TOTAL_WEIGHTS_COUNT)) {
        return false;
      }
      if (new_round != nullptr) {
        *new_round = fl_latest_round;
      }
      return true;
    }
    delay(20);
  }

  Serial.println("Timeout waiting for newer global model");
  return false;
}