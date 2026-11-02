import json
import os
import subprocess
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

import numpy as np
import requests
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# CONFIGURATION
# ============================================================

PROMETHEUS_URL = "http://localhost:9090"
MODEL_FILE = "model/live_adaptive_lstm.keras"
DEPLOYMENT_NAME = "predictive-app"
NAMESPACE = "default"

SEQUENCE_LENGTH = 12
HISTORY_HOURS = 3
STEP_SECONDS = 60

MIN_REPLICAS = 1
MAX_REPLICAS = 5

CONFIRMATION_COUNT = 3
COOLDOWN_SECONDS = 300
STATE_FILE = "scaling/scaler_state.json"

CPU_WEIGHT = 0.50
MEMORY_WEIGHT = 0.30
NETWORK_WEIGHT = 0.20

SCALE_UP_THRESHOLD = 0.65
SCALE_DOWN_THRESHOLD = 0.35

# ============================================================
# HEADER AND STATE MANAGEMENT
# ============================================================

print("=" * 70)
print("PREDICTIVE CONTAINER AUTO-SCALING")
print("=" * 70)


def load_state():
    default_state = {
        "last_decision": "KEEP",
        "confirmation_count": 0,
        "last_scale_time": 0,
    }

    if not os.path.exists(STATE_FILE):
        return default_state

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            saved_state = json.load(file)
        default_state.update(saved_state)
        return default_state
    except (OSError, ValueError, TypeError):
        print("Warning: Could not read scaler state. Using fresh state.")
        return default_state


def save_state(state):
    state_dir = os.path.dirname(STATE_FILE)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4)


state = load_state()

# ============================================================
# PROMETHEUS QUERY FUNCTION
# ============================================================


def query_prometheus_range(query, start, end, step):
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": start,
            "end": end,
            "step": step,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {data}")

    return data["data"]["result"]


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading adaptive LSTM model...")
model = tf.keras.models.load_model(MODEL_FILE)
print("Model loaded successfully.")
print(MODEL_FILE)

# ============================================================
# COLLECT RECENT WORKLOAD DATA
# ============================================================

print("\nCollecting recent Kubernetes workload data...")

end_time = int(time.time())
start_time = end_time - HISTORY_HOURS * 60 * 60

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

print("Querying CPU metrics...")
cpu_result = query_prometheus_range(
    cpu_query, start_time, end_time, STEP_SECONDS
)

print("Querying memory metrics...")
memory_result = query_prometheus_range(
    memory_query, start_time, end_time, STEP_SECONDS
)

print("Querying network metrics...")
network_result = query_prometheus_range(
    network_query, start_time, end_time, STEP_SECONDS
)

# ============================================================
# VALIDATE PROMETHEUS RESULTS
# ============================================================

if not cpu_result:
    raise RuntimeError("No CPU data received from Prometheus.")
if not memory_result:
    raise RuntimeError("No memory data received from Prometheus.")
if not network_result:
    raise RuntimeError("No network data received from Prometheus.")

# The queries aggregate matching containers, so normally each
# query returns one series.
cpu_values = cpu_result[0]["values"]
memory_values = memory_result[0]["values"]
network_values = network_result[0]["values"]


def to_timestamp_dict(values):
    result = {}
    for timestamp, value in values:
        try:
            number = float(value)
            if np.isfinite(number):
                result[int(float(timestamp))] = number
        except (TypeError, ValueError):
            continue
    return result


cpu_dict = to_timestamp_dict(cpu_values)
memory_dict = to_timestamp_dict(memory_values)
network_dict = to_timestamp_dict(network_values)

print("\nPrometheus sample counts:")
print("CPU timestamps:", len(cpu_dict))
print("Memory timestamps:", len(memory_dict))
print("Network timestamps:", len(network_dict))

# ============================================================
# ALIGN METRICS BY EXACT TIMESTAMP
# ============================================================

common_timestamps = sorted(
    set(cpu_dict.keys()) & set(memory_dict.keys()) & set(network_dict.keys())
)
print("Common timestamps:", len(common_timestamps))

if len(common_timestamps) < SEQUENCE_LENGTH:
    raise RuntimeError(
        f"Not enough aligned live samples. Need at least {SEQUENCE_LENGTH} "
        f"for the LSTM input, but received {len(common_timestamps)}. "
        "Check Prometheus query results and metric timestamps."
    )

live_data = np.array(
    [
        [
            cpu_dict[timestamp],
            memory_dict[timestamp],
            network_dict[timestamp],
        ]
        for timestamp in common_timestamps
    ],
    dtype=np.float32,
)

if not np.all(np.isfinite(live_data)):
    raise RuntimeError("Live workload data contains invalid values.")

print("\n" + "=" * 70)
print("CURRENT KUBERNETES WORKLOAD")
print("=" * 70)
print("Aligned samples:", len(live_data))
print(f"CPU     : {live_data[-1, 0]:.4f} %")
print(f"Memory  : {live_data[-1, 1]:.2f} KB")
print(f"Network : {live_data[-1, 2]:.4f} KB/s")

# ============================================================
# SCALE LIVE DATA
# ============================================================

print("\nCreating live calibration scaler...")
live_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_live_data = live_scaler.fit_transform(live_data)

latest_sequence = scaled_live_data[-SEQUENCE_LENGTH:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)
print("LSTM input shape:", latest_sequence.shape)

# ============================================================
# PREDICT RESOURCE USAGE
# ============================================================

print("\nPredicting future resource usage...")
prediction_scaled = model.predict(latest_sequence, verbose=0)

if prediction_scaled.shape != (1, 3):
    raise RuntimeError(
        f"Unexpected model output shape: {prediction_scaled.shape}; "
        "expected (1, 3)."
    )

prediction_scaled = np.clip(prediction_scaled, 0.0, 1.0)
prediction = live_scaler.inverse_transform(prediction_scaled)
prediction = np.maximum(prediction, 0.0)

print("\n" + "=" * 70)
print("LSTM PREDICTED RESOURCE USAGE")
print("=" * 70)
print(f"Predicted CPU     : {prediction[0, 0]:.4f} %")
print(f"Predicted Memory  : {prediction[0, 1]:.2f} KB")
print(f"Predicted Network : {prediction[0, 2]:.4f} KB/s")

# ============================================================
# MULTI-RESOURCE SCALING PRESSURE
# ============================================================

cpu_score = float(prediction_scaled[0, 0])
memory_score = float(prediction_scaled[0, 1])
network_score = float(prediction_scaled[0, 2])

pressure = (
    CPU_WEIGHT * cpu_score
    + MEMORY_WEIGHT * memory_score
    + NETWORK_WEIGHT * network_score
)

print("\n" + "=" * 70)
print("MULTI-RESOURCE PRESSURE")
print("=" * 70)
print(f"CPU score     : {cpu_score:.4f}")
print(f"Memory score  : {memory_score:.4f}")
print(f"Network score : {network_score:.4f}")
print(f"Combined pressure: {pressure:.4f}")

# ============================================================
# GET CURRENT REPLICAS
# ============================================================


def get_current_replicas():
    result = subprocess.run(
        [
            "kubectl",
            "get",
            "deployment",
            DEPLOYMENT_NAME,
            "-n",
            NAMESPACE,
            "-o",
            "jsonpath={.spec.replicas}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


current_replicas = get_current_replicas()
print("\nCurrent replicas:", current_replicas)

# ============================================================
# SCALING DECISION
# ============================================================

if pressure >= SCALE_UP_THRESHOLD:
    decision = "SCALE UP"
elif pressure <= SCALE_DOWN_THRESHOLD:
    decision = "SCALE DOWN"
else:
    decision = "KEEP"

print("\nPredictive scaling decision:", decision)

# ============================================================
# CONFIRMATION AND COOLDOWN
# ============================================================

current_time = time.time()
last_scale_time = float(state.get("last_scale_time", 0))
cooldown_active = (
    last_scale_time > 0
    and current_time - last_scale_time < COOLDOWN_SECONDS
)

previous_decision = state.get("last_decision", "KEEP")
confirmation_count = int(state.get("confirmation_count", 0))

if decision == "KEEP":
    confirmation_count = 0
    previous_decision = "KEEP"
elif decision == previous_decision:
    # Cap at the required count so it never displays 4/3, 5/3, etc.
    confirmation_count = min(
        confirmation_count + 1, CONFIRMATION_COUNT
    )
else:
    confirmation_count = 1
    previous_decision = decision

print(f"Confirmation: {confirmation_count}/{CONFIRMATION_COUNT}")
print("Cooldown active:", cooldown_active)

target_replicas = current_replicas
scaling_action_taken = False
confirmation_cycle_complete = confirmation_count >= CONFIRMATION_COUNT

if (
    decision == "SCALE UP"
    and confirmation_cycle_complete
    and not cooldown_active
    and current_replicas < MAX_REPLICAS
):
    target_replicas = min(current_replicas + 1, MAX_REPLICAS)

elif (
    decision == "SCALE DOWN"
    and confirmation_cycle_complete
    and not cooldown_active
    and current_replicas > MIN_REPLICAS
):
    target_replicas = max(current_replicas - 1, MIN_REPLICAS)

# ============================================================
# APPLY KUBERNETES SCALING
# ============================================================

if target_replicas != current_replicas:
    print(
        f"\nScaling {DEPLOYMENT_NAME} from "
        f"{current_replicas} to {target_replicas} replicas..."
    )

    subprocess.run(
        [
            "kubectl",
            "scale",
            "deployment",
            DEPLOYMENT_NAME,
            "-n",
            NAMESPACE,
            f"--replicas={target_replicas}",
        ],
        check=True,
    )

    state["last_decision"] = "KEEP"
    state["confirmation_count"] = 0
    state["last_scale_time"] = time.time()
    scaling_action_taken = True
    print("Kubernetes scaling command executed successfully.")

else:
    print("\nNO REPLICA CHANGE")
    print("Replica count remains:", current_replicas)

    # Preserve the last real scaling timestamp: a new run must not
    # reset or bypass the 300-second cooldown.
    state["last_scale_time"] = last_scale_time

    if decision == "KEEP":
        state["last_decision"] = "KEEP"
        state["confirmation_count"] = 0
    elif confirmation_cycle_complete and not cooldown_active:
        # A decision cycle has completed but no scaling was possible
        # (for example, SCALE DOWN at the minimum of 1 replica).
        # Reset only the confirmation cycle so the next run starts at 1/3.
        state["last_decision"] = "KEEP"
        state["confirmation_count"] = 0
    else:
        # Retain the current confirmation while waiting for cooldown,
        # so the action can occur once cooldown expires if the decision
        # still matches.
        state["last_decision"] = previous_decision
        state["confirmation_count"] = confirmation_count

save_state(state)

print("\n" + "=" * 70)
print("PREDICTIVE AUTO-SCALING COMPLETED")
print("=" * 70)
print(
    "Prometheus -> LSTM Prediction -> Multi-Resource Pressure "
    "-> Scaling Decision -> Kubernetes"
)
print("Final replica count:", target_replicas)
