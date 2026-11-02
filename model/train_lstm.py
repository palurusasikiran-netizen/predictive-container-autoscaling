import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path


# ============================================
# 1. LOAD PREPARED DATA
# ============================================

DATA_DIR = Path("dataset/processed")
MODEL_DIR = Path("model")

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")

X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")


print("Data loaded successfully!")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================
# 2. CREATE LSTM MODEL
# ============================================

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(3)
])


# ============================================
# 3. COMPILE MODEL
# ============================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\n========================================")
print("LSTM MODEL ARCHITECTURE")
print("========================================")

model.summary()


# ============================================
# 4. CALLBACKS
# ============================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = ModelCheckpoint(
    MODEL_DIR / "best_lstm.keras",
    monitor="val_loss",
    save_best_only=True
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================
# 5. TRAIN MODEL
# ============================================

print("\n========================================")
print("STARTING LSTM TRAINING")
print("========================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)


# ============================================
# 6. SAVE FINAL MODEL
# ============================================

model.save(MODEL_DIR / "lstm_model.keras")


print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")

print("Best model saved to:")
print(MODEL_DIR / "best_lstm.keras")

print("Final model saved to:")
print(MODEL_DIR / "lstm_model.keras")

print("\nTraining epochs completed:", len(history.history["loss"]))