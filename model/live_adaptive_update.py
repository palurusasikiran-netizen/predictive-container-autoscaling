import time
from pathlib import Path

import requests
import numpy as np
import joblib
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

PROMETHEUS_URL = "http://localhost:9090"

BASE_MODEL_FILE = "model/best_lstm.keras"
ADAPTIVE_MODEL_FILE = "model/live_adaptive_lstm.keras"
SCALER_FILE = "dataset/processed/scalers.pkl"

SEQUENCE_LENGTH = 12

# Prometheus sampling interval
STEP_SECONDS = 300          # 5 minutes

# Recent workload history
HISTORY_HOURS = 3           # 3 hours

# Adaptive learning parameters
LEARNING_RATE = 0.0001
EPOCHS = 3
BATCH_SIZE = 8


# ============================================================
# START
# ============================================================

print("=" * 60)
print("LIVE ADAPTIVE LSTM UPDATE")
print("=" * 60)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not Path(BASE_MODEL_FILE).exists():
    raise FileNotFoundError(
        f"Base model not found: {BASE_MODEL_FILE}"
    )

if not Path(SCALER_FILE).exists():
    raise FileNotFoundError(
        f"Scaler file not found: {SCALER_FILE}"
    )


# ============================================================
# LOAD TRAINED LSTM MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = tf.keras.models.load_model(BASE_MODEL_FILE)

print("Base model loaded successfully.")
print("Model:", BASE_MODEL_FILE)


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scalers = joblib.load(SCALER_FILE)

# VM-1 scaler used during original training
scaler = scalers[1]

print("Scaler loaded successfully.")
print("Using VM-1 scaler.")


# ============================================================
# QUERY PROMETHEUS RANGE
# ============================================================

def query_prometheus_range(query, start, end, step):
    """
    Query Prometheus for historical time-series data.
    """

    url = f"{PROMETHEUS_URL}/api/v1/query_range"

    response = requests.get(
        url,
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
# DEFINE TIME RANGE
# ============================================================

end_time = int(time.time())

start_time = end_time - (
    HISTORY_HOURS * 60 * 60
)

print("\nCollecting recent workload data from Prometheus...")

print(f"History window    : {HISTORY_HOURS} hours")
print(f"Sampling interval : {STEP_SECONDS} seconds")


# ============================================================
# CPU QUERY
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


# ============================================================
# MEMORY QUERY
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


# ============================================================
# NETWORK QUERY
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


# ============================================================
# COLLECT CPU DATA
# ============================================================

print("\nQuerying CPU metrics...")

cpu_result = query_prometheus_range(
    cpu_query,
    start_time,
    end_time,
    STEP_SECONDS
)


# ============================================================
# COLLECT MEMORY DATA
# ============================================================

print("Querying memory metrics...")

memory_result = query_prometheus_range(
    memory_query,
    start_time,
    end_time,
    STEP_SECONDS
)


# ============================================================
# COLLECT NETWORK DATA
# ============================================================

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
    raise RuntimeError(
        "No CPU data received from Prometheus."
    )

if not memory_result:
    raise RuntimeError(
        "No memory data received from Prometheus."
    )

if not network_result:
    raise RuntimeError(
        "No network data received from Prometheus."
    )


# ============================================================
# EXTRACT TIME-SERIES VALUES
# ============================================================

cpu_values = cpu_result[0]["values"]

memory_values = memory_result[0]["values"]

network_values = network_result[0]["values"]


print("\nPrometheus data received successfully.")

print("CPU samples     :", len(cpu_values))
print("Memory samples  :", len(memory_values))
print("Network samples :", len(network_values))


# ============================================================
# ALIGN DATA USING TIMESTAMPS
# ============================================================

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


common_timestamps = sorted(
    set(cpu_dict.keys())
    & set(memory_dict.keys())
    & set(network_dict.keys())
)


# ============================================================
# CHECK ALIGNED DATA
# ============================================================

minimum_required = SEQUENCE_LENGTH + 1

if len(common_timestamps) < minimum_required:
    raise RuntimeError(
        f"Not enough aligned Prometheus samples.\n"
        f"Required : {minimum_required}\n"
        f"Available: {len(common_timestamps)}\n\n"
        f"Make sure Prometheus has been collecting metrics "
        f"for at least {SEQUENCE_LENGTH + 1} samples."
    )


print("\nAligned timestamps:", len(common_timestamps))


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
# DISPLAY RECENT LIVE WORKLOAD
# ============================================================

print("\n" + "=" * 46)
print("RECENT LIVE WORKLOAD DATA")
print("=" * 46)

print(
    "Aligned samples:",
    len(live_data)
)

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
# SCALE LIVE DATA
# ============================================================

print("\nScaling live workload data...")

scaled_data = scaler.transform(live_data)


# ============================================================
# CREATE LSTM TRAINING SEQUENCES
# ============================================================

X_live = []
y_live = []


for i in range(
    len(scaled_data) - SEQUENCE_LENGTH
):

    X_live.append(
        scaled_data[
            i:i + SEQUENCE_LENGTH
        ]
    )

    y_live.append(
        scaled_data[
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


# ============================================================
# DISPLAY TRAINING DATA
# ============================================================

print("\n" + "=" * 46)
print("LIVE ADAPTIVE TRAINING DATA")
print("=" * 46)

print(
    "X_live shape:",
    X_live.shape
)

print(
    "y_live shape:",
    y_live.shape
)


# ============================================================
# PREPARE MODEL FOR ADAPTIVE LEARNING
# ============================================================

print("\nPreparing model for adaptive learning...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mse",
    metrics=["mae"]
)


print("\nModel recompiled successfully.")

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Epochs        :",
    EPOCHS
)

print(
    "Batch size    :",
    BATCH_SIZE
)


# ============================================================
# EVALUATE BEFORE ADAPTIVE UPDATE
# ============================================================

print("\nEvaluating model before adaptive update...")

before_loss, before_mae = model.evaluate(
    X_live,
    y_live,
    verbose=0
)

print(
    f"Before update - Loss: {before_loss:.8f}"
)

print(
    f"Before update - MAE : {before_mae:.8f}"
)


# ============================================================
# START ADAPTIVE TRAINING
# ============================================================

print("\n" + "=" * 46)
print("STARTING LIVE ADAPTIVE UPDATE")
print("=" * 46)


history = model.fit(
    X_live,
    y_live,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1
)


# ============================================================
# EVALUATE AFTER ADAPTIVE UPDATE
# ============================================================

print("\nEvaluating model after adaptive update...")

after_loss, after_mae = model.evaluate(
    X_live,
    y_live,
    verbose=0
)


# ============================================================
# SAVE UPDATED MODEL
# ============================================================

model.save(
    ADAPTIVE_MODEL_FILE
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LIVE ADAPTIVE UPDATE COMPLETED")
print("=" * 60)

print("\nUpdated model saved to:")

print(
    ADAPTIVE_MODEL_FILE
)


print("\n" + "=" * 60)
print("ADAPTIVE LEARNING RESULTS")
print("=" * 60)

print(
    f"Before update Loss : {before_loss:.8f}"
)

print(
    f"After update Loss  : {after_loss:.8f}"
)

print(
    f"Before update MAE  : {before_mae:.8f}"
)

print(
    f"After update MAE   : {after_mae:.8f}"
)


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

if before_mae != 0:

    mae_improvement = (
        (before_mae - after_mae)
        / before_mae
    ) * 100

else:

    mae_improvement = 0.0


print(
    f"\nMAE change: {mae_improvement:.2f}%"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("LIVE ADAPTIVE COMPONENT STATUS")
print("=" * 60)


if after_mae < before_mae:

    print(
        "Model adapted successfully "
        "to recent Kubernetes workload patterns."
    )

else:

    print(
        "Adaptive update completed."
    )

    print(
        "Recent workload did not reduce "
        "the training MAE."
    )


print("\nAdaptive model:")

print(
    ADAPTIVE_MODEL_FILE
)

print("\nLive adaptive learning completed successfully!")