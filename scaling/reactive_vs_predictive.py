import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "dataset/processed/test_predictions.csv"

OUTPUT_FILE = "scaling/reactive_vs_predictive_results.csv"

GRAPH_FILE = "graphs/reactive_vs_predictive.png"

CPU_THRESHOLD = 65.0


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================

print("=" * 65)
print("REACTIVE VS PREDICTIVE AUTO-SCALING COMPARISON")
print("=" * 65)

print("\nLoading test prediction data...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Total samples:", len(df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "VM_ID",
    "Actual_CPU_%",
    "Predicted_CPU_%"
]

for column in required_columns:

    if column not in df.columns:
        raise RuntimeError(
            f"Required column missing: {column}"
        )


# ============================================================
# CONVERT NORMALIZED CPU VALUES TO PERCENTAGE
# ============================================================

print("\nConverting normalized CPU values to percentage...")

df["Actual_CPU_Percent"] = df["Actual_CPU_%"]
df["Predicted_CPU_Percent"] = df["Predicted_CPU_%"]

print("CPU conversion completed.")


# ============================================================
# REACTIVE SCALING DECISION
# ============================================================

df["Reactive_Decision"] = "KEEP"

df.loc[
    df["Actual_CPU_Percent"] >= CPU_THRESHOLD,
    "Reactive_Decision"
] = "SCALE UP"


# ============================================================
# PREDICTIVE SCALING DECISION
# ============================================================

df["Predictive_Decision"] = "KEEP"

df.loc[
    df["Predicted_CPU_Percent"] >= CPU_THRESHOLD,
    "Predictive_Decision"
] = "SCALE UP"


# ============================================================
# COUNT SCALE-UP DECISIONS
# ============================================================

reactive_scaleups = (
    df["Reactive_Decision"] == "SCALE UP"
).sum()

predictive_scaleups = (
    df["Predictive_Decision"] == "SCALE UP"
).sum()


# ============================================================
# COUNT KEEP DECISIONS
# ============================================================

reactive_keep = (
    df["Reactive_Decision"] == "KEEP"
).sum()

predictive_keep = (
    df["Predictive_Decision"] == "KEEP"
).sum()


# ============================================================
# DECISION AGREEMENT
# ============================================================

agreement = (
    df["Reactive_Decision"]
    ==
    df["Predictive_Decision"]
).sum()

agreement_percentage = (
    agreement / len(df)
) * 100


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("COMPARISON RESULTS")
print("=" * 65)

print(
    f"\nCPU scale-up threshold : "
    f"{CPU_THRESHOLD:.1f}%"
)

print("\nREACTIVE AUTO-SCALING")

print(
    f"Scale-up decisions : "
    f"{reactive_scaleups}"
)

print(
    f"Keep decisions      : "
    f"{reactive_keep}"
)

print("\nPREDICTIVE AUTO-SCALING")

print(
    f"Scale-up decisions : "
    f"{predictive_scaleups}"
)

print(
    f"Keep decisions      : "
    f"{predictive_keep}"
)

print("\nDECISION AGREEMENT")

print(
    f"Matching decisions : "
    f"{agreement} / {len(df)}"
)

print(
    f"Agreement           : "
    f"{agreement_percentage:.2f}%"
)


# ============================================================
# DISPLAY SAMPLE COMPARISON
# ============================================================

print("\n" + "=" * 65)
print("SAMPLE COMPARISON")
print("=" * 65)

sample = df[
    [
        "VM_ID",
        "Actual_CPU_Percent",
        "Predicted_CPU_Percent",
        "Reactive_Decision",
        "Predictive_Decision"
    ]
].head(20)

print(
    sample.to_string(index=False)
)


# ============================================================
# CREATE COMPARISON GRAPH
# ============================================================

print("\nCreating comparison graph...")

plt.figure(figsize=(12, 6))

plt.plot(
    df["Actual_CPU_Percent"].head(100),
    label="Actual CPU"
)

plt.plot(
    df["Predicted_CPU_Percent"].head(100),
    label="Predicted CPU"
)

plt.axhline(
    CPU_THRESHOLD,
    linestyle="--",
    label="CPU Threshold"
)

plt.xlabel("Test Sample")

plt.ylabel("CPU Usage (%)")

plt.title(
    "Reactive vs Predictive Auto-Scaling"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_FILE,
    dpi=300
)

plt.close()


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 65)
print("REACTIVE VS PREDICTIVE COMPARISON COMPLETED")
print("=" * 65)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nGraph saved to:")
print(GRAPH_FILE)

print("\nComparison pipeline:")
print("Actual CPU -> Reactive scaling decision")
print("LSTM predicted CPU -> Predictive scaling decision")

print("\nDone.")