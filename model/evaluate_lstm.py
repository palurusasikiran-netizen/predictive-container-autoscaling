import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path


# ============================================
# 1. PATHS
# ============================================

DATA_DIR = Path("dataset/processed")
MODEL_FILE = Path("model/best_lstm.keras")

SCALER_FILE = DATA_DIR / "scalers.pkl"
VM_IDS_FILE = DATA_DIR / "test_vm_ids.npy"

PREDICTION_FILE = DATA_DIR / "test_predictions.csv"


# ============================================
# 2. LOAD TEST DATA
# ============================================

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")
test_vm_ids = np.load(VM_IDS_FILE)

print("Test data loaded successfully!")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("test_vm_ids shape:", test_vm_ids.shape)


# ============================================
# 3. LOAD SCALERS
# ============================================

scalers = joblib.load(SCALER_FILE)

print("\nScalers loaded successfully!")
print("Number of VM scalers:", len(scalers))


# ============================================
# 4. LOAD TRAINED MODEL
# ============================================

model = tf.keras.models.load_model(MODEL_FILE)

print("\nModel loaded successfully!")
print("Model:", MODEL_FILE)


# ============================================
# 5. MAKE PREDICTIONS
# ============================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test,
    batch_size=128,
    verbose=1
)

print("\nPrediction completed!")
print("Prediction shape:", y_pred.shape)


# ============================================
# 6. NORMALIZED-SCALE EVALUATION
# ============================================

resource_names = [
    "CPU",
    "Memory",
    "Network"
]

print("\n========================================")
print("NORMALIZED MODEL EVALUATION")
print("========================================")

normalized_results = []

for i, resource in enumerate(resource_names):

    mae = mean_absolute_error(
        y_test[:, i],
        y_pred[:, i]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test[:, i],
            y_pred[:, i]
        )
    )

    normalized_results.append({
        "Resource": resource,
        "MAE": mae,
        "RMSE": rmse
    })

    print(f"\n{resource}:")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")


# ============================================
# 7. CONVERT PREDICTIONS BACK TO ORIGINAL UNITS
# ============================================

print("\n========================================")
print("CONVERTING TO ORIGINAL UNITS")
print("========================================")

y_test_original = np.zeros_like(y_test)
y_pred_original = np.zeros_like(y_pred)

unique_vms = np.unique(test_vm_ids)

for vm_id in unique_vms:

    vm_mask = test_vm_ids == vm_id

    scaler = scalers[int(vm_id)]

    y_test_original[vm_mask] = scaler.inverse_transform(
        y_test[vm_mask]
    )

    y_pred_original[vm_mask] = scaler.inverse_transform(
        y_pred[vm_mask]
    )

print("Conversion completed successfully.")


# ============================================
# 8. ORIGINAL-UNIT EVALUATION
# ============================================

print("\n========================================")
print("ORIGINAL-UNIT MODEL EVALUATION")
print("========================================")

original_results = []

units = [
    "%",
    "KB",
    "KB/s"
]

for i, resource in enumerate(resource_names):

    mae = mean_absolute_error(
        y_test_original[:, i],
        y_pred_original[:, i]
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_original[:, i],
            y_pred_original[:, i]
        )
    )

    original_results.append({
        "Resource": resource,
        "Unit": units[i],
        "MAE": mae,
        "RMSE": rmse
    })

    print(f"\n{resource} ({units[i]}):")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")


# ============================================
# 9. SAVE PREDICTIONS
# ============================================

prediction_df = pd.DataFrame({
    "VM_ID": test_vm_ids,
    "Actual_CPU_%": y_test_original[:, 0],
    "Predicted_CPU_%": y_pred_original[:, 0],
    "Actual_Memory_KB": y_test_original[:, 1],
    "Predicted_Memory_KB": y_pred_original[:, 1],
    "Actual_Network_KBps": y_test_original[:, 2],
    "Predicted_Network_KBps": y_pred_original[:, 2]
})

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print("\n========================================")
print("PREDICTIONS SAVED")
print("========================================")

print("Saved to:", PREDICTION_FILE)


# ============================================
# 10. SAMPLE PREDICTIONS
# ============================================

print("\n========================================")
print("SAMPLE PREDICTIONS")
print("========================================")

print(
    prediction_df.head(10).to_string(index=False)
)


# ============================================
# 11. OVERALL NORMALIZED PERFORMANCE
# ============================================

overall_mae = mean_absolute_error(
    y_test,
    y_pred
)

overall_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("\n========================================")
print("OVERALL NORMALIZED PERFORMANCE")
print("========================================")

print(f"Overall MAE  : {overall_mae:.6f}")
print(f"Overall RMSE : {overall_rmse:.6f}")


# ============================================
# 12. EVALUATION COMPLETED
# ============================================

print("\n========================================")
print("LSTM EVALUATION COMPLETED SUCCESSFULLY")
print("========================================")