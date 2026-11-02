
import pandas as pd
from pathlib import Path

for file in sorted(Path("dataset/raw").glob("*.csv"),
                   key=lambda p: int(p.stem)):
    df = pd.read_csv(file, sep="\t")
    df.columns = df.columns.str.strip().str.rstrip(";")

    for col in [
        "Timestamp [ms]",
        "CPU usage [%]",
        "Memory usage [KB]",
        "Network received throughput [KB/s]",
        "Network transmitted throughput [KB/s]",
    ]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.strip().str.rstrip(";"),
            errors="coerce"
        )

    df["Network throughput [KB/s]"] = (
        df["Network received throughput [KB/s]"]
        + df["Network transmitted throughput [KB/s]"]
    )

    cols = [
        "Timestamp [ms]",
        "CPU usage [%]",
        "Memory usage [KB]",
        "Network throughput [KB/s]",
    ]

    exact_duplicate_rows = df.duplicated(cols).sum()
    unique_timestamps = df["Timestamp [ms]"].nunique()
    conflicting_groups = (
        df.groupby("Timestamp [ms]")[cols[1:]]
        .apply(lambda g: len(g.drop_duplicates()) > 1)
        .sum()
    )

    print(
        f"VM {file.stem}: rows={len(df)}, "
        f"exact_duplicate_rows={exact_duplicate_rows}, "
        f"unique_timestamps={unique_timestamps}, "
        f"conflicting_timestamp_groups={conflicting_groups}"
    )
