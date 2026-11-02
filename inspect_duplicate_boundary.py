
import pandas as pd

for vm in [2, 4, 5, 6, 7, 8, 9, 10]:
    path = f"dataset/raw/{vm}.csv"
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip().str.rstrip(";")

    ts = pd.to_numeric(
        df["Timestamp [ms]"].astype(str).str.strip().str.rstrip(";"),
        errors="coerce"
    )

    repeated_times = ts[ts.duplicated(keep=False)]
    first_repeat = repeated_times.min()
    positions = df.index[ts == first_repeat].tolist()

    print(f"\nVM {vm}: first repeated timestamp = {first_repeat}")
    print("Row positions:", positions)

    start = max(0, positions[0] - 3)
    end = min(len(df), positions[-1] + 4)

    print(
        df.iloc[start:end][
            ["Timestamp [ms]", "CPU usage [%]",
             "Memory usage [KB]",
             "Network received throughput [KB/s]",
             "Network transmitted throughput [KB/s]"]
        ].to_string(index=True)
    )
