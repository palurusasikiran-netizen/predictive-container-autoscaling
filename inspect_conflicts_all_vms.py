
import pandas as pd
from pathlib import Path

features = [
    "CPU usage [%]",
    "Memory usage [KB]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]

for file in sorted(Path("dataset/raw").glob("*.csv"),
                   key=lambda p: int(p.stem)):
    df = pd.read_csv(file, sep="\t")
    df.columns = df.columns.str.strip().str.rstrip(";")

    for col in ["Timestamp [ms]"] + features:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.rstrip(";"),
            errors="coerce"
        )

    distinct = df.drop_duplicates(
        subset=["Timestamp [ms]"] + features
    )

    counts = distinct.groupby("Timestamp [ms]").size()
    conflict_times = counts[counts > 1].index

    if len(conflict_times):
        print(f"\n--- VM {file.stem}: conflicting timestamps ---")
        print(
            distinct[distinct["Timestamp [ms]"].isin(conflict_times)]
            [["Timestamp [ms]"] + features]
            .sort_values("Timestamp [ms]")
            .to_string(index=False)
        )
