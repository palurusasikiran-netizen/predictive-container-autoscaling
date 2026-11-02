import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path


# ============================================
# 1. PATHS
# ============================================

DATA_DIR = Path("dataset/processed")

BASE_MODEL_FILE = Path("model/best_lstm.keras")

ADAPTIVE_MODEL_FILE = Path("model/adaptive_lstm.keras")


# ============================================
# 2. LOAD TEST DATA
# ============================================

print("========================================")
print("BASE LSTM vs ADAPTIVE LSTM")
print("========================================")

print("\nLoading untouched test data...")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("Test data loaded successfully!")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================
# 3. LOAD BASE MODEL
# ============================================

print("\nLoading base LSTM model...")

base_model = tf.keras.models.load_model(BASE_MODEL_FILE)

print("Base model loaded successfully!")


# ============================================
# 4. LOAD ADAPTIVE MODEL
# ============================================

print("\nLoading adaptive LSTM model...")

adaptive_model = tf.keras.models.load_model(
    ADAPTIVE_MODEL_FILE
)

print("Adaptive model loaded successfully!")


# ============================================
# 5. GENERATE PREDICTIONS
# ============================================

print("\nGenerating predictions using base model...")

base_predictions = base_model.predict(
    X_test,
    batch_size=128,
    verbose=1
)

print("\nGenerating predictions using adaptive model...")

adaptive_predictions = adaptive_model.predict(
    X_test,
    batch_size=128,
    verbose=1
)


# ============================================
# 6. RESOURCE NAMES
# ============================================

resource_names = [
    "CPU",
    "Memory",
    "Network"
]


# ============================================
# 7. CALCULATE METRICS
# ============================================

base_mae_values = []
base_rmse_values = []

adaptive_mae_values = []
adaptive_rmse_values = []


print("\n========================================")
print("MODEL COMPARISON RESULTS")
print("========================================")


for i, resource in enumerate(resource_names):

    base_mae = mean_absolute_error(
        y_test[:, i],
        base_predictions[:, i]
    )

    base_rmse = np.sqrt(
        mean_squared_error(
            y_test[:, i],
            base_predictions[:, i]
        )
    )

    adaptive_mae = mean_absolute_error(
        y_test[:, i],
        adaptive_predictions[:, i]
    )

    adaptive_rmse = np.sqrt(
        mean_squared_error(
            y_test[:, i],
            adaptive_predictions[:, i]
        )
    )

    base_mae_values.append(base_mae)
    base_rmse_values.append(base_rmse)

    adaptive_mae_values.append(adaptive_mae)
    adaptive_rmse_values.append(adaptive_rmse)

    print(f"\n{resource}:")

    print("Base LSTM:")
    print(f"  MAE  : {base_mae:.6f}")
    print(f"  RMSE : {base_rmse:.6f}")

    print("Adaptive LSTM:")
    print(f"  MAE  : {adaptive_mae:.6f}")
    print(f"  RMSE : {adaptive_rmse:.6f}")


# ============================================
# 8. OVERALL METRICS
# ============================================

base_overall_mae = mean_absolute_error(
    y_test,
    base_predictions
)

base_overall_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        base_predictions
    )
)

adaptive_overall_mae = mean_absolute_error(
    y_test,
    adaptive_predictions
)

adaptive_overall_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        adaptive_predictions
    )
)


# ============================================
# 9. IMPROVEMENT CALCULATION
# ============================================

mae_improvement = (
    (base_overall_mae - adaptive_overall_mae)
    / base_overall_mae
) * 100

rmse_improvement = (
    (base_overall_rmse - adaptive_overall_rmse)
    / base_overall_rmse
) * 100


# ============================================
# 10. DISPLAY OVERALL COMPARISON
# ============================================

print("\n========================================")
print("OVERALL MODEL COMPARISON")
print("========================================")

print("\nBase LSTM:")
print(f"Overall MAE  : {base_overall_mae:.6f}")
print(f"Overall RMSE : {base_overall_rmse:.6f}")

print("\nAdaptive LSTM:")
print(f"Overall MAE  : {adaptive_overall_mae:.6f}")
print(f"Overall RMSE : {adaptive_overall_rmse:.6f}")

print("\n========================================")
print("ADAPTIVE LEARNING IMPROVEMENT")
print("========================================")

print(f"MAE improvement  : {mae_improvement:.2f}%")
print(f"RMSE improvement : {rmse_improvement:.2f}%")


# ============================================
# 11. FINAL CONCLUSION
# ============================================

print("\n========================================")
print("COMPARISON COMPLETED")
print("========================================")

if adaptive_overall_mae < base_overall_mae:

    print(
        "Adaptive LSTM achieved lower MAE "
        "than the base LSTM."
    )

else:

    print(
        "Adaptive LSTM did not achieve lower MAE "
        "on the current test data."
    )


if adaptive_overall_rmse < base_overall_rmse:

    print(
        "Adaptive LSTM achieved lower RMSE "
        "than the base LSTM."
    )

else:

    print(
        "Adaptive LSTM did not achieve lower RMSE "
        "on the current test data."
    )

print("\nBase model:")
print(BASE_MODEL_FILE)

print("\nAdaptive model:")
print(ADAPTIVE_MODEL_FILE)