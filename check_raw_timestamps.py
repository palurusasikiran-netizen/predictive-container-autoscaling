
import pandas as pd
from pathlib import Path

raw_dir = Path("dataset/raw")

for file in sorted(raw_dir.glob("*.csv"), key=lambda p: int(p.stem)):
    df = pd.read_csv(file, sep="\t")

    df.columns = df.columns.str.strip().str.rstrip(";")

    timestamps = pd.to_numeric(
        df["Timestamp [ms]"].astype(str).str.strip().str.rstrip(";"),
        errors="coerce"
    )

    print(
        f"{file.name}: rows={len(df)}, "
        f"duplicate timestamps={timestamps.duplicated().sum()}, "
        f"unique timestamps={timestamps.nunique()}"
    )

print("\nFirst 12 timestamps from raw VM 2:")

df = pd.read_csv(raw_dir / "2.csv", sep="\t")
df.columns = df.columns.str.strip().str.rstrip(";")

print(df["Timestamp [ms]"].head(12).to_string(index=False))
