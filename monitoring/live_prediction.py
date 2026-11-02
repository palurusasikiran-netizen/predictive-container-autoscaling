import requests
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROMETHEUS_URL = "http://localhost:9090"

MODEL_FILE = "model/live_adaptive_lstm.keras"

SEQUENCE_LENGTH = 12

HISTORY_HOURS = 3

STEP_SECONDS = 300


# ============================================================
# LOAD ADAPTIVE LSTM MODEL
# ============================================================

print("=" * 60)
print("LIVE KUBERNETES LSTM PREDICTION")
print("=" * 60)

print("\nLoading adaptive LSTM model...")

model = tf.keras.models.load_model(MODEL_FILE)

print("Model loaded successfully:")
print(MODEL_FILE)


# ============================================================
# PROMETHEUS RANGE QUERY
# ============================================================

def query_prometheus_range(query, start, end, step):

    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": start,
            "end": end,
            "step": step
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if data["status"] != "success":
        raise RuntimeError(data)

    return data["data"]["result"]


# ============================================================
# TIME RANGE
# ============================================================

import time

end_time = int(time.time())

start_time = end_time - (HISTORY_HOURS * 60 * 60)

print("\nCollecting recent Kubernetes workload data...")

print(f"History window : {HISTORY_HOURS} hours")
print(f"Sampling step  : {STEP_SECONDS} seconds")


# ============================================================
# CPU
# ============================================================

cpu_query = """
sum(
    rate(
        container_cpu_usage_seconds_total{
            namespace="default",
            pod=~"predictive-app-.*",
            container!="POD",
            container!=""
        }[5m]
    )
) * 100
"""

print("\nQuerying CPU metrics...")

cpu_result = query_prometheus_range(
    cpu_query,
    start_time,
    end_time,
    STEP_SECONDS
)


# ============================================================
# MEMORY
# ============================================================

memory_query = """
sum(
    container_memory_working_set_bytes{
        namespace="default",
        pod=~"predictive-app-.*",
        container!="POD",
        container!=""
    }
) / 1024
"""

print("Querying memory metrics...")

memory_result = query_prometheus_range(
    memory_query,
    start_time,
    end_time,
    STEP_SECONDS
)


# ============================================================
# NETWORK
# ============================================================

network_query = """
(
    sum(
        rate(
            container_network_receive_bytes_total{
                namespace="default",
                pod=~"predictive-app-.*"
            }[5m]
        )
    )
    +
    sum(
        rate(
            container_network_transmit_bytes_total{
                namespace="default",
                pod=~"predictive-app-.*"
            }[5m]
        )
    )
) / 1024
"""

print("Querying network metrics...")

network_result = query_prometheus_range(
    network_query,
    start_time,
    end_time,
    STEP_SECONDS
)


# ============================================================
# CHECK PROMETHEUS DATA
# ============================================================

if not cpu_result:
    raise RuntimeError("No CPU data received from Prometheus.")

if not memory_result:
    raise RuntimeError("No memory data received from Prometheus.")

if not network_result:
    raise RuntimeError("No network data received from Prometheus.")


# ============================================================
# CONVERT PROMETHEUS RESULTS TO DICTIONARIES
# ============================================================

cpu_values = cpu_result[0]["values"]

memory_values = memory_result[0]["values"]

network_values = network_result[0]["values"]


cpu_dict = {
    int(float(timestamp)): float(value)
    for timestamp, value in cpu_values
}

memory_dict = {
    int(float(timestamp)): float(value)
    for timestamp, value in memory_values
}

network_dict = {
    int(float(timestamp)): float(value)
    for timestamp, value in network_values
}


# ============================================================
# ALIGN TIMESTAMPS
# ============================================================

common_timestamps = sorted(
    set(cpu_dict.keys())
    &
    set(memory_dict.keys())
    &
    set(network_dict.keys())
)


if len(common_timestamps) < SEQUENCE_LENGTH + 1:

    raise RuntimeError(
        f"Not enough live data. "
        f"Need at least {SEQUENCE_LENGTH + 1} samples, "
        f"but only {len(common_timestamps)} are available."
    )


# ============================================================
# CREATE LIVE WORKLOAD DATASET
# ============================================================

live_data = []

for timestamp in common_timestamps:

    cpu = cpu_dict[timestamp]

    memory_kb = memory_dict[timestamp]

    network_kbps = network_dict[timestamp]

    live_data.append(
        [
            cpu,
            memory_kb,
            network_kbps
        ]
    )


live_data = np.array(
    live_data,
    dtype=np.float32
)


# ============================================================
# DISPLAY LIVE WORKLOAD
# ============================================================

print("\n" + "=" * 60)
print("RECENT KUBERNETES WORKLOAD")
print("=" * 60)

print("Aligned samples :", len(live_data))

print("\nLatest observation:")

print(
    f"CPU     : {live_data[-1][0]:.4f} %"
)

print(
    f"Memory  : {live_data[-1][1]:.2f} KB"
)

print(
    f"Network : {live_data[-1][2]:.4f} KB/s"
)


# ============================================================
# LIVE CALIBRATION SCALER
# ============================================================
#
# The original LSTM was trained on Bitbrains VM telemetry.
# Kubernetes telemetry has a different workload distribution.
#
# For this Review-2 live-integration prototype, we calibrate
# the recent Kubernetes workload into a 0-1 range.
#
# This prevents the old VM scaler from forcing Kubernetes
# values to inappropriate training extremes.
# ============================================================

print("\nCreating live Kubernetes calibration scaler...")

live_scaler = MinMaxScaler(
    feature_range=(0, 1)
)

scaled_live_data = live_scaler.fit_transform(
    live_data
)


print("Live calibration completed.")


# ============================================================
# CREATE LSTM SEQUENCES
# ============================================================

X_live = []

y_live = []


for i in range(
    len(scaled_live_data) - SEQUENCE_LENGTH
):

    X_live.append(
        scaled_live_data[
            i:i + SEQUENCE_LENGTH
        ]
    )

    y_live.append(
        scaled_live_data[
            i + SEQUENCE_LENGTH
        ]
    )


X_live = np.array(
    X_live,
    dtype=np.float32
)

y_live = np.array(
    y_live,
    dtype=np.float32
)


print("\n" + "=" * 60)
print("LIVE LSTM DATA")
print("=" * 60)

print("X_live shape:", X_live.shape)

print("y_live shape:", y_live.shape)


# ============================================================
# USE MOST RECENT SEQUENCE FOR PREDICTION
# ============================================================

latest_sequence = scaled_live_data[
    -SEQUENCE_LENGTH:
]

latest_sequence = np.expand_dims(
    latest_sequence,
    axis=0
)


print(
    "\nLSTM input shape:",
    latest_sequence.shape
)


# ============================================================
# PREDICT NEXT RESOURCE USAGE
# ============================================================

print("\nPredicting next resource usage...")

prediction_scaled = model.predict(
    latest_sequence,
    verbose=0
)


# ============================================================
# KEEP PREDICTION INSIDE LIVE CALIBRATION RANGE
# ============================================================

prediction_scaled = np.clip(
    prediction_scaled,
    0.0,
    1.0
)


# ============================================================
# CONVERT BACK TO KUBERNETES RESOURCE UNITS
# ============================================================

prediction = live_scaler.inverse_transform(
    prediction_scaled
)


prediction = np.maximum(
    prediction,
    0.0
)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("LSTM PREDICTED NEXT RESOURCE USAGE")
print("=" * 60)

print(
    f"Predicted CPU     : "
    f"{prediction[0][0]:.4f} %"
)

print(
    f"Predicted Memory  : "
    f"{prediction[0][1]:.2f} KB"
)

print(
    f"Predicted Network : "
    f"{prediction[0][2]:.4f} KB/s"
)


# ============================================================
# CURRENT VS PREDICTED
# ============================================================

print("\n" + "=" * 60)
print("CURRENT VS PREDICTED")
print("=" * 60)

print(
    f"CPU     : "
    f"{live_data[-1][0]:.4f} %"
    f"  ->  "
    f"{prediction[0][0]:.4f} %"
)

print(
    f"Memory  : "
    f"{live_data[-1][1]:.2f} KB"
    f"  ->  "
    f"{prediction[0][1]:.2f} KB"
)

print(
    f"Network : "
    f"{live_data[-1][2]:.4f} KB/s"
    f"  ->  "
    f"{prediction[0][2]:.4f} KB/s"
)


# ============================================================
# RESOURCE CHANGE
# ============================================================

cpu_change = (
    prediction[0][0]
    - live_data[-1][0]
)

memory_change = (
    prediction[0][1]
    - live_data[-1][1]
)

network_change = (
    prediction[0][2]
    - live_data[-1][2]
)


print("\n" + "=" * 60)
print("PREDICTED RESOURCE CHANGE")
print("=" * 60)

print(
    f"CPU change     : {cpu_change:+.4f} %"
)

print(
    f"Memory change  : {memory_change:+.2f} KB"
)

print(
    f"Network change : {network_change:+.4f} KB/s"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("LIVE PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nPrometheus -> Kubernetes metrics -> "
    "LSTM -> Resource prediction"
)