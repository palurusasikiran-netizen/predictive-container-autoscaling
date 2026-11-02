
import pandas as pd
from pathlib import Path

for file in sorted(
    Path("dataset/raw").glob("*.csv"),
    key=lambda p: int(p.stem)
):
    df = pd.read_csv(file, sep="\t")
    df.columns = df.columns.str.strip().str.rstrip(";")

    ts = pd.to_numeric(
        df["Timestamp [ms]"].astype(str).str.strip().str.rstrip(";"),
        errors="coerce"
    )

    repeated = ts[ts.duplicated(keep=False)]

    print(f"\nVM {file.stem}")
    print("First timestamp:", ts.min())
    print("Last timestamp:", ts.max())
    print("Rows with repeated timestamps:", len(repeated))

    if not repeated.empty:
        print("First repeated timestamp:", repeated.min())
        print("Last repeated timestamp:", repeated.max())
