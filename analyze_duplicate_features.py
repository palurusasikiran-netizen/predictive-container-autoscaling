
import pandas as pd

df = pd.read_csv("dataset/raw/2.csv", sep="\t")
df.columns = df.columns.str.strip().str.rstrip(";")

for col in df.columns:
    df[col] = df[col].astype(str).str.strip().str.rstrip(";")

df["Timestamp [ms]"] = pd.to_numeric(
    df["Timestamp [ms]"], errors="coerce"
)

features = [
    "CPU usage [%]",
    "Memory usage [KB]",
    "Network received throughput [KB/s]",
    "Network transmitted throughput [KB/s]",
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Network throughput [KB/s]"] = (
    df["Network received throughput [KB/s]"]
    + df["Network transmitted throughput [KB/s]"]
)

model_features = [
    "CPU usage [%]",
    "Memory usage [KB]",
    "Network throughput [KB/s]",
]

repeated = df[df.duplicated("Timestamp [ms]", keep=False)]

print("Repeated-timestamp rows:", len(repeated))
print(
    "Rows identical across timestamp and model features:",
    repeated.duplicated(
        ["Timestamp [ms]"] + model_features
    ).sum()
)

print("\nUnique feature combinations per repeated timestamp:")
print(
    repeated.groupby("Timestamp [ms]")[model_features]
    .apply(lambda g: len(g.drop_duplicates()))
    .value_counts()
    .sort_index()
    .to_string()
)
