"""
Copyright (C) 2026  Fraunhofer Institute for Microelectronic Circuits and Systems.
All rights reserved.
AIfES x Flower Demonstrator is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import os
import signal
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import paho.mqtt.client as mqtt
import torch
import torch.nn as nn
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

# Data type used for all model weights
DTYPE = np.float32

# Model architecture configuration
INPUTS = 60
DENSE_NEURONS_1 = 4
DENSE_NEURONS_2 = 3

# Minimum number of clients required for aggregation
MIN_CLIENTS = 2

# =========================
# MQTT CONFIG (STATIC)
# =========================

# MQTT broker connection settings
MQTT_BROKER_HOST = 'broker.hivemq.com'
MQTT_BROKER_PORT = 1883
MQTT_USERNAME = ''
MQTT_PASSWORD = ''
MQTT_TOPIC_PREFIX = 'demo/fl'
MQTT_KEEPALIVE_S = 60

# MQTT topics used by the federated learning server
TOPIC_GLOBAL_LATEST = f'{MQTT_TOPIC_PREFIX}/global/latest'
TOPIC_SERVER_HEALTH = f'{MQTT_TOPIC_PREFIX}/server/health'
TOPIC_CLIENT_UPDATE_WILDCARD = f'{MQTT_TOPIC_PREFIX}/clients/+/update'
TOPIC_CLIENT_ACK_FMT = f'{MQTT_TOPIC_PREFIX}/clients/{{client_id}}/ack'

# Binary packet headers for global model messages and client update messages
GLOBAL_PACKET_HEADER = struct.Struct('<II')
UPDATE_PACKET_HEADER = struct.Struct('<III')


class TinyMLNet(nn.Module):
    """
    Defines a small neural network for TinyML classification tasks.

    The model consists of two fully connected layers. The first layer
    applies a sigmoid activation function, and the second layer applies
    a softmax function to produce class probabilities.
    """

    def __init__(self) -> None:
        """
        Initializes the TinyML network with two dense layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(INPUTS, DENSE_NEURONS_1, bias=True)
        self.fc2 = nn.Linear(DENSE_NEURONS_1, DENSE_NEURONS_2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        :param x: Input tensor
        :return: Output tensor containing softmax probabilities
        """
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=-1)
        return x


class DummyClientProxy:
    """
    Represents a minimal client proxy for Flower aggregation.

    This helper class is used to provide a client identifier in a format
    compatible with the Flower strategy interface.
    """

    def __init__(self, cid: str) -> None:
        """
        Initializes the dummy client proxy.

        :param cid: Unique client identifier
        """
        self.cid = cid


@dataclass
class AckMessage:
    """
    Represents an acknowledgment message sent back to a client.

    The message includes status information, details about the operation,
    the current federated learning round, and update statistics.
    """

    status: str
    detail: str
    round: int
    received_updates: int
    min_clients: int

    def to_json(self) -> str:
        """
        Converts the acknowledgment message to a JSON string.

        :return: JSON representation of the acknowledgment message
        """
        return json.dumps(
            {
                'status': self.status,
                'detail': self.detail,
                'round': self.round,
                'received_updates': self.received_updates,
                'min_clients': self.min_clients,
            }
        )


class FLState:
    """
    Stores and manages the state of the federated learning server.

    This includes the current round, global model weights, the minimum
    number of required clients, pending client updates, and the Flower
    aggregation strategy.
    """

    def __init__(self, initial_weights: np.ndarray, min_clients: int) -> None:
        """
        Initializes the federated learning state.

        :param initial_weights: Initial global model weights
        :param min_clients: Minimum number of clients required for aggregation
        """
        self.lock = threading.Lock()
        self.weight_count = int(initial_weights.size)
        self.min_clients = min_clients
        self.strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            min_evaluate_clients=0,
            accept_failures=False,
        )
        self.current_round = 0
        self.global_weights = initial_weights.astype(DTYPE, copy=True)
        self.updates: Dict[str, Tuple[np.ndarray, int]] = {}

    def serialize_global_packet(self, round_number: int, weights: np.ndarray) -> bytes:
        """
        Serializes the global model weights into a binary packet.

        The packet contains the round number, the number of weights,
        and the flattened weight values.

        :param round_number: Current federated learning round
        :param weights: Global model weights
        :return: Serialized binary packet
        """
        weights = np.asarray(weights, dtype=DTYPE).reshape(-1)
        return GLOBAL_PACKET_HEADER.pack(round_number, weights.size) + weights.tobytes(order='C')

    def parse_update_packet(self, raw: bytes) -> Tuple[int, int, np.ndarray]:
        """
        Parses a binary client update packet.

        The packet contains the client round number, the number of weights,
        the number of local training examples, and the weight values.

        :param raw: Raw binary MQTT payload
        :return: Tuple containing round number, number of examples, and weights
        :raises ValueError: If the packet size or contents are invalid
        """
        if len(raw) < UPDATE_PACKET_HEADER.size:
            raise ValueError('payload too small')

        round_number, count, num_examples = UPDATE_PACKET_HEADER.unpack_from(raw, 0)
        if count != self.weight_count:
            raise ValueError(f'unexpected weight count: got {count}, expected {self.weight_count}')
        if num_examples <= 0:
            raise ValueError('num_examples must be > 0')

        expected = UPDATE_PACKET_HEADER.size + count * 4
        if len(raw) != expected:
            raise ValueError(f'unexpected payload size: got {len(raw)}, expected {expected}')

        weights = np.frombuffer(raw[UPDATE_PACKET_HEADER.size :], dtype=DTYPE).copy()
        return int(round_number), int(num_examples), weights

    def aggregate_if_ready(self) -> bool:
        """
        Aggregates all currently stored client updates using the FedAvg strategy.

        The method converts the stored updates into Flower FitRes objects,
        performs federated averaging, updates the global model weights,
        increments the current round, and clears the pending updates.

        :return: True if aggregation was performed successfully
        :raises RuntimeError: If aggregation fails or returns invalid results
        """
        # Add this condition if synchronous aggregation should be enforced.
        # if len(self.updates) < self.min_clients:
        #     return False

        results = []
        for cid, (weights, num_examples) in self.updates.items():
            fit_res = FitRes(
                status=Status(code=Code.OK, message='OK'),
                parameters=ndarrays_to_parameters([weights]),
                num_examples=num_examples,
                metrics={},
            )
            results.append((DummyClientProxy(cid), fit_res))

        aggregated_parameters, _ = self.strategy.aggregate_fit(
            server_round=self.current_round + 1,
            results=results,
            failures=[],
        )
        if aggregated_parameters is None:
            raise RuntimeError('aggregation failed')

        aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        if len(aggregated_ndarrays) != 1:
            raise RuntimeError('unexpected number of parameter tensors')

        aggregated = np.asarray(aggregated_ndarrays[0], dtype=DTYPE).reshape(-1)
        if aggregated.size != self.weight_count:
            raise RuntimeError('aggregated weight count mismatch')

        self.global_weights = aggregated.copy()
        self.current_round += 1
        self.updates.clear()
        return True


# --- model helpers ---------------------------------------------------------


def create_initial_model() -> TinyMLNet:
    """
    Creates the initial TinyML model with a fixed random seed.

    :return: Initialized TinyMLNet model
    """
    torch.manual_seed(42)
    return TinyMLNet()


def flatten_model_weights(model: TinyMLNet) -> np.ndarray:
    """
    Flattens all model weights and biases into a single NumPy array.

    The order is:
    - fc1 weights
    - fc1 bias
    - fc2 weights
    - fc2 bias

    :param model: TinyML model to extract weights from
    :return: Flattened NumPy array containing all model parameters
    """
    parts = [
        model.fc1.weight.detach().cpu().numpy().astype(DTYPE).reshape(-1),
        model.fc1.bias.detach().cpu().numpy().astype(DTYPE).reshape(-1),
        model.fc2.weight.detach().cpu().numpy().astype(DTYPE).reshape(-1),
        model.fc2.bias.detach().cpu().numpy().astype(DTYPE).reshape(-1),
    ]
    return np.concatenate(parts, axis=0)


# Create the initial model instance
INITIAL_MODEL = create_initial_model()

# Extract the initial model weights as a flat array
INITIAL_WEIGHTS = flatten_model_weights(INITIAL_MODEL)

# Create the federated learning server state
state = FLState(initial_weights=INITIAL_WEIGHTS, min_clients=MIN_CLIENTS)

# Create the MQTT client used by the server
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id='fl-server-mqtt')

# Event used to stop the server loop gracefully
stop_event = threading.Event()


def publish_health() -> None:
    """
    Publishes the current health status of the server via MQTT.

    The published payload contains the current round, the minimum number
    of clients, the number of received updates, the model weight count,
    and basic model configuration information.
    """
    payload = json.dumps(
        {
            'status': 'ok',
            'round': state.current_round,
            'min_clients': state.min_clients,
            'received_updates': len(state.updates),
            'weight_count': state.weight_count,
            'model': {
                'inputs': INPUTS,
                'dense_1': DENSE_NEURONS_1,
                'dense_2': DENSE_NEURONS_2,
            },
        }
    )
    mqtt_client.publish(TOPIC_SERVER_HEALTH, payload=payload, qos=1, retain=True)


def publish_latest_global() -> None:
    """
    Publishes the latest global model weights via MQTT.

    The global model is serialized into a binary packet and published
    to the configured global topic. After that, the current health
    status is also published.
    """
    packet = state.serialize_global_packet(state.current_round, state.global_weights)
    mqtt_client.publish(TOPIC_GLOBAL_LATEST, payload=packet, qos=1, retain=True)
    publish_health()


def publish_ack(client_id: str, ack: AckMessage) -> None:
    """
    Publishes an acknowledgment message to a specific client.

    :param client_id: Identifier of the client receiving the acknowledgment
    :param ack: Acknowledgment message object
    """
    mqtt_client.publish(
        TOPIC_CLIENT_ACK_FMT.format(client_id=client_id),
        payload=ack.to_json(),
        qos=1,
        retain=False,
    )


def extract_client_id(topic: str) -> Optional[str]:
    """
    Extracts the client identifier from an MQTT topic.

    The expected topic format is:
    <topic_prefix>/clients/<client_id>/update

    :param topic: MQTT topic string
    :return: Extracted client identifier, or None if the topic is invalid
    """
    parts = topic.split('/')
    prefix_parts = MQTT_TOPIC_PREFIX.split('/')
    if len(parts) < len(prefix_parts) + 3:
        return None
    if parts[: len(prefix_parts)] != prefix_parts:
        return None
    if parts[len(prefix_parts)] != 'clients':
        return None
    return parts[len(prefix_parts) + 1]


def on_connect(client: mqtt.Client, userdata, flags, reason_code, properties) -> None:
    """
    Callback function called after a successful MQTT connection.

    The server subscribes to all client update topics and immediately
    publishes the latest global model.

    :param client: MQTT client instance
    :param userdata: User-defined callback data
    :param flags: Response flags sent by the broker
    :param reason_code: MQTT connection result code
    :param properties: MQTT connection properties
    """
    print(f'Connected to MQTT broker: reason_code={reason_code}')
    client.subscribe(TOPIC_CLIENT_UPDATE_WILDCARD, qos=1)
    publish_latest_global()


def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage) -> None:
    """
    Callback function called when an MQTT message is received.

    This function validates incoming client updates, checks the round number,
    stores the update, optionally triggers model aggregation, and sends an
    acknowledgment back to the client.

    :param client: MQTT client instance
    :param userdata: User-defined callback data
    :param msg: Received MQTT message
    """
    client_id = extract_client_id(msg.topic)
    if not client_id or not msg.topic.endswith('/update'):
        return

    try:
        payload_round, num_examples, weights = state.parse_update_packet(msg.payload)
    except ValueError as exc:
        publish_ack(
            client_id,
            AckMessage(
                'error', str(exc), state.current_round, len(state.updates), state.min_clients
            ),
        )
        return

    with state.lock:
        if payload_round != state.current_round:
            publish_ack(
                client_id,
                AckMessage(
                    'error',
                    f'stale round: client={payload_round}, server={state.current_round}',
                    state.current_round,
                    len(state.updates),
                    state.min_clients,
                ),
            )
            return

        state.updates[client_id] = (weights, num_examples)
        aggregated = False
        try:
            aggregated = state.aggregate_if_ready()
        except Exception as exc:
            publish_ack(
                client_id,
                AckMessage(
                    'error', str(exc), state.current_round, len(state.updates), state.min_clients
                ),
            )
            return

        publish_ack(
            client_id,
            AckMessage(
                'accepted',
                'update stored' if not aggregated else 'update stored and aggregated',
                state.current_round,
                len(state.updates),
                state.min_clients,
            ),
        )
        publish_health()
        if aggregated:
            publish_latest_global()
            print(f'Aggregated new global round {state.current_round}')


# Register MQTT callback functions
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Set MQTT authentication if credentials are provided
if MQTT_USERNAME:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)


def handle_signal(signum, frame) -> None:
    """
    Handles termination signals for graceful shutdown.

    When a SIGINT or SIGTERM signal is received, the stop event is set
    so that the main loop can terminate cleanly.

    :param signum: Signal number
    :param frame: Current stack frame
    """
    stop_event.set()


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


if __name__ == '__main__':
    """
    Starts the MQTT-based federated learning server.

    The server connects to the configured MQTT broker, starts the MQTT loop,
    and waits for incoming client updates until a termination signal is received.
    """
    print(f'Connecting to MQTT broker {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}')
    print(f'Topic prefix: {MQTT_TOPIC_PREFIX}')
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=MQTT_KEEPALIVE_S)
    mqtt_client.loop_start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
