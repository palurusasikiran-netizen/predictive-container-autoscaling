
import pandas as pd
from pathlib import Path

raw_dir = Path("dataset/raw")

columns_to_check = [
    "CPU usage [%]",
    "CPU usage [MHZ]",
    "Memory usage [KB]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]"
]

files = sorted(raw_dir.glob("*.csv"), key=lambda p: int(p.stem))

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file.name)

    df = pd.read_csv(file, sep="\t", dtype=str)

    df.columns = df.columns.str.strip().str.rstrip(";")

    for column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.rstrip(";")
        )

    print("Available columns:", list(df.columns))

    for column in columns_to_check:
        if column not in df.columns:
            print(f"{column}: NOT FOUND")
            continue

        values = pd.to_numeric(df[column], errors="coerce")

        print(f"\n{column}")
        print("  Total rows:", len(values))
        print("  Zero values:", int((values == 0).sum()))
        print("  Missing/non-numeric:", int(values.isna().sum()))
        print("  Minimum:", values.min())
        print("  Maximum:", values.max())

