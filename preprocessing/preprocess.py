import pandas as pd
from pathlib import Path

RAW_DIR = Path("dataset/raw")
OUTPUT_FILE = Path("dataset/clean_dataset.csv")

dataframes = []

required_columns = [
    "Timestamp [ms]",
    "CPU usage [%]",
    "Memory usage [KB]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]"
]

for file in sorted(
    RAW_DIR.glob("*.csv"),
    key=lambda x: int(x.stem)
):
    print(f"Reading {file.name}...")

    df = pd.read_csv(file, sep="\t", dtype=str)

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.rstrip(";")
    )

    # Clean all values
    for column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.rstrip(";")
        )

    # Convert required columns to numeric
    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Calculate total network throughput
    df["Network throughput [KB/s]"] = (
        df["Network received throughput [KB/s]"]
        + df["Network transmitted throughput [KB/s]"]
    )

    # Keep only required project columns
    df = df[
        [
            "Timestamp [ms]",
            "CPU usage [%]",
            "Memory usage [KB]",
            "Network throughput [KB/s]"
        ]
    ]

    # Add VM identifier
    df["VM_ID"] = file.stem

    # Remove only rows where required values are missing
    df = df.dropna(
        subset=[
            "Timestamp [ms]",
            "CPU usage [%]",
            "Memory usage [KB]",
            "Network throughput [KB/s]"
        ]
    )

    print(f"  Valid rows: {len(df)}")

    dataframes.append(df)


# Combine all VMs
combined_df = pd.concat(
    dataframes,
    ignore_index=True
)

# Sort by VM and timestamp
combined_df = combined_df.sort_values(
    by=["VM_ID", "Timestamp [ms]"]
)

# Save cleaned dataset
combined_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("========================================")

print("Total VMs:", combined_df["VM_ID"].nunique())
print("Total rows:", len(combined_df))

print("\nFinal columns:")
print(combined_df.columns.tolist())

print("\nMissing values:")
print(combined_df.isnull().sum())

print("\nSample data:")
print(combined_df.head())

print("\nSaved to:")
print(OUTPUT_FILE)