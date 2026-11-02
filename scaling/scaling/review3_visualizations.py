
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

INPUT_FILE = "scaling/reactive_vs_predictive_results.csv"
GRAPH_DIR = "graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

required = [
    "Actual_CPU_Percent",
    "Predicted_CPU_Percent",
    "Reactive_Decision",
    "Predictive_Decision",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Graph 1: CPU values in a representative window.
# Use a middle window rather than assuming the first 100 samples
# represent the complete test set.
window_size = min(100, len(df))
start = max(0, (len(df) - window_size) // 2)
window = df.iloc[start:start + window_size]

plt.figure(figsize=(12, 5))
plt.plot(
    range(start, start + len(window)),
    window["Actual_CPU_Percent"],
    label="Actual CPU",
    linewidth=1.5,
)
plt.plot(
    range(start, start + len(window)),
    window["Predicted_CPU_Percent"],
    label="Predicted CPU",
    linewidth=1.5,
)
plt.axhline(
    65.0,
    linestyle="--",
    label="CPU threshold (65%)",
)
plt.xlabel("Test sample index")
plt.ylabel("CPU usage (%)")
plt.title("Actual vs Predicted CPU Usage — 100-Sample Window")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(GRAPH_DIR, "review3_cpu_window.png"),
    dpi=300,
)
plt.close()

# Graph 2: Decision confusion matrix.
labels = ["KEEP", "SCALE UP"]
matrix = confusion_matrix(
    df["Reactive_Decision"],
    df["Predictive_Decision"],
    labels=labels,
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels,
)
display.plot(values_format=",d")
plt.title("Predictive vs Reactive CPU-Threshold Decisions")
plt.tight_layout()
plt.savefig(
    os.path.join(GRAPH_DIR, "review3_decision_matrix.png"),
    dpi=300,
)
plt.close()

print("Visualization generation completed.")
print("CPU graph: graphs/review3_cpu_window.png")
print("Decision matrix: graphs/review3_decision_matrix.png")
print("Samples:", len(df))
print("\nDecision counts:")
print(pd.crosstab(
    df["Reactive_Decision"],
    df["Predictive_Decision"],
).to_string())
