
import pandas as pd

df = pd.read_csv("dataset/clean_dataset.csv")

print("VM | Rows | Unique timestamps | Median interval | Min | Max")

for vm, group in df.groupby("VM_ID"):
    timestamps = (
        group["Timestamp [ms]"]
        .drop_duplicates()
        .sort_values()
    )

    intervals = timestamps.diff().dropna()

    print(
        vm,
        "|", len(group),
        "|", timestamps.nunique(),
        "|", intervals.median(),
        "|", intervals.min(),
        "|", intervals.max()
    )
