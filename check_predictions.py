
import pandas as pd

path = "dataset/processed/test_predictions.csv"
df = pd.read_csv(path)

checks = {
    "CPU predictions below 0": df["Predicted_CPU_%"] < 0,
    "CPU predictions above 100": df["Predicted_CPU_%"] > 100,
    "Memory predictions below 0": df["Predicted_Memory_KB"] < 0,
    "Network predictions below 0": df["Predicted_Network_KBps"] < 0,
}

print("Prediction validity report")
print("Total test predictions:", len(df))

for label, mask in checks.items():
    print(f"{label}: {int(mask.sum())}")

print("\nPrediction ranges:")
for column in [
    "Predicted_CPU_%",
    "Predicted_Memory_KB",
    "Predicted_Network_KBps",
]:
    print(
        column,
        "| min:", df[column].min(),
        "| max:", df[column].max()
    )
