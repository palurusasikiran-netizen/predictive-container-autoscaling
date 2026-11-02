import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib


# ============================================
# 1. PATHS AND SETTINGS
# ============================================

INPUT_FILE = Path("dataset/clean_dataset.csv")
OUTPUT_DIR = Path("dataset/processed")

FEATURES = [
    "CPU usage [%]",
    "Memory usage [KB]",
    "Network throughput [KB/s]"
]

SEQUENCE_LENGTH = 12

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


# ============================================
# 2. CREATE TIME-SERIES SEQUENCES
# ============================================

def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(sequence_length, len(data)):

        X.append(
            data[i - sequence_length:i]
        )

        y.append(
            data[i]
        )

    return np.array(X), np.array(y)


# ============================================
# 3. LOAD CLEAN DATASET
# ============================================

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Total rows:", len(df))
print("Total VMs:", df["VM_ID"].nunique())


# ============================================
# 4. CREATE OUTPUT DIRECTORY
# ============================================

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)


# ============================================
# 5. STORAGE FOR ALL VMs
# ============================================

all_X_train = []
all_y_train = []

all_X_val = []
all_y_val = []

all_X_test = []
all_y_test = []

# NEW:
# Stores the VM ID corresponding to every
# test sequence.
all_test_vm_ids = []

# Store one scaler for each VM
scalers = {}


# ============================================
# 6. PROCESS EACH VM
# ============================================

for vm_id, vm_df in df.groupby("VM_ID"):

    print(f"\nProcessing VM {vm_id}...")

    # Sort according to timestamp
    vm_df = vm_df.sort_values(
        "Timestamp [ms]"
    ).reset_index(drop=True)

    # Select only the three resources
    values = vm_df[FEATURES].values

    # Number of rows for this VM
    n = len(values)

    # ========================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ========================================

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_values = values[:train_end]

    validation_values = values[
        train_end:validation_end
    ]

    test_values = values[
        validation_end:
    ]


    # ========================================
    # NORMALIZATION
    # ========================================

    scaler = MinMaxScaler()

    # IMPORTANT:
    # Fit scaler only on training data
    train_scaled = scaler.fit_transform(
        train_values
    )

    validation_scaled = scaler.transform(
        validation_values
    )

    test_scaled = scaler.transform(
        test_values
    )


    # Save scaler for this VM
    scalers[vm_id] = scaler


    # ========================================
    # CREATE SEQUENCES
    # ========================================

    X_train, y_train = create_sequences(
        train_scaled,
        SEQUENCE_LENGTH
    )

    X_val, y_val = create_sequences(
        validation_scaled,
        SEQUENCE_LENGTH
    )

    X_test, y_test = create_sequences(
        test_scaled,
        SEQUENCE_LENGTH
    )


    # ========================================
    # STORE DATA
    # ========================================

    all_X_train.append(X_train)
    all_y_train.append(y_train)

    all_X_val.append(X_val)
    all_y_val.append(y_val)

    all_X_test.append(X_test)
    all_y_test.append(y_test)


    # ========================================
    # STORE VM ID FOR EACH TEST SEQUENCE
    # ========================================

    vm_test_ids = np.full(
        len(y_test),
        vm_id
    )

    all_test_vm_ids.append(
        vm_test_ids
    )


    print(
        f"VM {vm_id}: "
        f"train={len(X_train)}, "
        f"validation={len(X_val)}, "
        f"test={len(X_test)}"
    )


# ============================================
# 7. COMBINE DATA FROM ALL VMs
# ============================================

X_train = np.concatenate(
    all_X_train
)

y_train = np.concatenate(
    all_y_train
)

X_val = np.concatenate(
    all_X_val
)

y_val = np.concatenate(
    all_y_val
)

X_test = np.concatenate(
    all_X_test
)

y_test = np.concatenate(
    all_y_test
)


# NEW:
# Combine VM IDs in exactly the same order
# as y_test
test_vm_ids = np.concatenate(
    all_test_vm_ids
)


# ============================================
# 8. SAVE PROCESSED DATA
# ============================================

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)


# NEW:
# Save VM ID corresponding to every test sample
np.save(
    OUTPUT_DIR / "test_vm_ids.npy",
    test_vm_ids
)


# ============================================
# 9. SAVE VM-SPECIFIC SCALERS
# ============================================

joblib.dump(
    scalers,
    OUTPUT_DIR / "scalers.pkl"
)


# ============================================
# 10. FINAL INFORMATION
# ============================================

print("\n========================================")
print("SEQUENCE PREPARATION COMPLETED")
print("========================================")

print(
    "X_train shape:",
    X_train.shape
)

print(
    "y_train shape:",
    y_train.shape
)

print(
    "X_val shape:",
    X_val.shape
)

print(
    "y_val shape:",
    y_val.shape
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)

print(
    "test_vm_ids shape:",
    test_vm_ids.shape
)

print(
    "\nFeatures:",
    FEATURES
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "History represented: 60 minutes"
)

print(
    "\nNumber of saved scalers:",
    len(scalers)
)

print(
    "\nSaved processed data to:"
)

print(
    OUTPUT_DIR
)