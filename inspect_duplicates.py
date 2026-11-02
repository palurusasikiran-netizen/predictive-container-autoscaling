
import pandas as pd

df = pd.read_csv("dataset/raw/2.csv", sep="\t")
df.columns = df.columns.str.strip().str.rstrip(";")

df["Timestamp [ms]"] = pd.to_numeric(
    df["Timestamp [ms]"].astype(str).str.strip().str.rstrip(";"),
    errors="coerce"
)

counts = df["Timestamp [ms]"].value_counts()

print("Timestamp repetition counts:")
print(counts.value_counts().sort_index().to_string())

print("\nExact duplicate rows:", df.duplicated().sum())

print("\nFirst timestamps appearing more than once:")
repeated = counts[counts > 1].index[:10]

print(
    df[df["Timestamp [ms]"].isin(repeated)]
    .sort_values("Timestamp [ms]")
    .to_string(index=False)
)
