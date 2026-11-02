
import pandas as pd
from pathlib import Path

for vm in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    file = Path(f"dataset/raw/{vm}.csv")
    df = pd.read_csv(file, sep="\t")
    df.columns = df.columns.str.strip().str.rstrip(";")

    ts = pd.to_numeric(
        df["Timestamp [ms]"].astype(str).str.strip().str.rstrip(";"),
        errors="coerce"
    )

    counts = ts.value_counts()

    print(
        f"VM {vm}: "
        f"rows={len(df)}, "
        f"unique_timestamps={ts.nunique()}, "
        f"timestamps occurring twice={(counts == 2).sum()}, "
        f"timestamps occurring 3+ times={(counts >= 3).sum()}"
    )

print("\nFirst 20 timestamps from VM 2:")
print(ts.head(20).to_string(index=False))
