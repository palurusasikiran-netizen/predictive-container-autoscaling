import numpy as np
import tensorflow as tf
from pathlib import Path


# ============================================
# 1. PATHS
# ============================================

DATA_DIR = Path("dataset/processed")

BASE_MODEL_FILE = Path("model/best_lstm.keras")

ADAPTIVE_MODEL_FILE = Path("model/adaptive_lstm.keras")


# ============================================
# 2. ADAPTIVE LEARNING SETTINGS
# ============================================

ADAPTIVE_EPOCHS = 5

ADAPTIVE_BATCH_SIZE = 128

NEW_DATA_RATIO = 0.20


# ============================================
# 3. LOAD EXISTING MODEL
# ============================================

print("========================================")
print("ADAPTIVE LSTM MODEL UPDATE")
print("========================================")

print("\nLoading existing trained model...")

model = tf.keras.models.load_model(BASE_MODEL_FILE)

print("Base model loaded successfully!")
print("Base model:", BASE_MODEL_FILE)


# ============================================
# 4. LOAD NEW WORKLOAD DATA
# ============================================

print("\nLoading newly observed workload data...")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

print("Validation data loaded successfully!")

print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)


# ============================================
# 5. SELECT RECENT WORKLOAD DATA
# ============================================

total_samples = len(X_val)

new_data_size = int(total_samples * NEW_DATA_RATIO)

X_new = X_val[-new_data_size:]

y_new = y_val[-new_data_size:]

print("\n========================================")
print("NEW WORKLOAD DATA")
print("========================================")

print("Total available validation samples:", total_samples)

print("Samples used for adaptive update:", len(X_new))

print("New data ratio:", NEW_DATA_RATIO)


# ============================================
# 6. RECOMPILE MODEL WITH LOW LEARNING RATE
# ============================================

print("\nPreparing model for adaptive learning...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="mse",
    metrics=["mae"]
)

print("Model recompiled successfully!")

print("Learning rate: 0.0001")


# ============================================
# 7. ADAPTIVE MODEL UPDATE
# ============================================

print("\n========================================")
print("STARTING ADAPTIVE MODEL UPDATE")
print("========================================")

history = model.fit(
    X_new,
    y_new,
    epochs=ADAPTIVE_EPOCHS,
    batch_size=ADAPTIVE_BATCH_SIZE,
    shuffle=False,
    verbose=1
)


# ============================================
# 8. SAVE UPDATED MODEL
# ============================================

model.save(ADAPTIVE_MODEL_FILE)

print("\n========================================")
print("ADAPTIVE UPDATE COMPLETED")
print("========================================")

print("Updated model saved to:")
print(ADAPTIVE_MODEL_FILE)


# ============================================
# 9. DISPLAY TRAINING RESULTS
# ============================================

print("\n========================================")
print("ADAPTIVE LEARNING RESULTS")
print("========================================")

print("Initial adaptive-update loss:",
      f"{history.history['loss'][0]:.6f}")

print("Final adaptive-update loss:",
      f"{history.history['loss'][-1]:.6f}")

print("Initial adaptive-update MAE:",
      f"{history.history['mae'][0]:.6f}")

print("Final adaptive-update MAE:",
      f"{history.history['mae'][-1]:.6f}")


# ============================================
# 10. COMPLETION MESSAGE
# ============================================

print("\n========================================")
print("ADAPTIVE LSTM COMPONENT READY")
print("========================================")

print("The trained LSTM has been updated")
print("using newly observed workload patterns.")

print("\nAdaptive model:")
print(ADAPTIVE_MODEL_FILE)