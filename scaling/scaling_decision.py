import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from pathlib import Path


# ============================================
# 1. PATHS
# ============================================

DATA_DIR = Path("dataset/processed")
MODEL_FILE = Path("model/best_lstm.keras")
OUTPUT_FILE = DATA_DIR / "scaling_decisions.csv"


# ============================================
# 2. RESOURCE WEIGHTS
# ============================================

CPU_WEIGHT = 0.50
MEMORY_WEIGHT = 0.30
NETWORK_WEIGHT = 0.20


# ============================================
# 3. SCALING THRESHOLDS
# ============================================

SCALE_UP_THRESHOLD = 0.65
SCALE_DOWN_THRESHOLD = 0.35


# ============================================
# 4. LOAD TEST DATA
# ============================================

X_test = np.load(DATA_DIR / "X_test.npy")
test_vm_ids = np.load(DATA_DIR / "test_vm_ids.npy")

print("Test data loaded successfully!")
print("X_test shape:", X_test.shape)
print("VM ID shape:", test_vm_ids.shape)


# ============================================
# 5. LOAD TRAINED MODEL
# ============================================

model = tf.keras.models.load_model(MODEL_FILE)

print("\nModel loaded successfully!")
print("Model:", MODEL_FILE)


# ============================================
# 6. LOAD SCALERS
# ============================================

scalers = joblib.load(DATA_DIR / "scalers.pkl")

print("\nScalers loaded successfully!")
print("Number of VM scalers:", len(scalers))


# ============================================
# 7. GENERATE NORMALIZED PREDICTIONS
# ============================================

print("\nGenerating normalized predictions...")

y_pred = model.predict(
    X_test,
    batch_size=128,
    verbose=1
)

print("\nPrediction completed!")
print("Prediction shape:", y_pred.shape)


# ============================================
# 8. CREATE RESULT DATAFRAME
# ============================================

results = []

for i in range(len(y_pred)):

    vm_id = int(test_vm_ids[i])

    cpu = float(y_pred[i][0])
    memory = float(y_pred[i][1])
    network = float(y_pred[i][2])

    # Keep values inside the expected normalized range
    cpu = np.clip(cpu, 0, 1)
    memory = np.clip(memory, 0, 1)
    network = np.clip(network, 0, 1)

    # ========================================
    # Multi-resource pressure
    # ========================================

    pressure = (
        CPU_WEIGHT * cpu
        + MEMORY_WEIGHT * memory
        + NETWORK_WEIGHT * network
    )

    # ========================================
    # Scaling decision
    # ========================================

    if pressure >= SCALE_UP_THRESHOLD:
        decision = "SCALE UP"

    elif pressure <= SCALE_DOWN_THRESHOLD:
        decision = "SCALE DOWN"

    else:
        decision = "KEEP"

    results.append(
        [
            vm_id,
            cpu,
            memory,
            network,
            pressure,
            decision
        ]
    )


# ============================================
# 9. CREATE DATAFRAME
# ============================================

result_df = pd.DataFrame(
    results,
    columns=[
        "VM_ID",
        "Predicted_CPU_Normalized",
        "Predicted_Memory_Normalized",
        "Predicted_Network_Normalized",
        "Resource_Pressure",
        "Scaling_Decision"
    ]
)


# ============================================
# 10. DISPLAY RESULTS
# ============================================

print("\n========================================")
print("MULTI-RESOURCE SCALING DECISIONS")
print("========================================")

print(
    result_df.head(20).to_string(index=False)
)


# ============================================
# 11. DECISION SUMMARY
# ============================================

print("\n========================================")
print("DECISION SUMMARY")
print("========================================")

decision_counts = result_df["Scaling_Decision"].value_counts()

print(
    "SCALE UP  :",
    decision_counts.get("SCALE UP", 0)
)

print(
    "KEEP      :",
    decision_counts.get("KEEP", 0)
)

print(
    "SCALE DOWN:",
    decision_counts.get("SCALE DOWN", 0)
)


# ============================================
# 12. SAVE RESULTS
# ============================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("SCALING DECISION COMPLETED")
print("========================================")

print("Saved to:", OUTPUT_FILE)