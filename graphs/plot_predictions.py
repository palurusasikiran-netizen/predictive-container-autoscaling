import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================
# 1. PATHS
# ============================================

INPUT_FILE = Path("dataset/processed/test_predictions.csv")
OUTPUT_DIR = Path("graphs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 2. LOAD PREDICTIONS
# ============================================

df = pd.read_csv(INPUT_FILE)

print("Prediction data loaded successfully!")
print("Total prediction rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# ============================================
# 3. SELECT DATA FOR PLOTTING
# ============================================

# Use the first 200 test samples for a clear graph.
# The complete prediction data remains available
# in test_predictions.csv.

plot_df = df.head(200).copy()

print("\nSamples used for graphs:", len(plot_df))


# ============================================
# 4. CPU ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(
    plot_df["Actual_CPU_%"],
    label="Actual CPU"
)

plt.plot(
    plot_df["Predicted_CPU_%"],
    label="Predicted CPU"
)

plt.xlabel("Test Sample")
plt.ylabel("CPU Usage (%)")
plt.title("Actual vs Predicted CPU Usage")
plt.legend()
plt.grid(True)
plt.tight_layout()

cpu_file = OUTPUT_DIR / "cpu_actual_vs_predicted.png"

plt.savefig(cpu_file, dpi=300)
plt.close()

print("\nCPU graph saved to:")
print(cpu_file)


# ============================================
# 5. MEMORY ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(
    plot_df["Actual_Memory_KB"],
    label="Actual Memory"
)

plt.plot(
    plot_df["Predicted_Memory_KB"],
    label="Predicted Memory"
)

plt.xlabel("Test Sample")
plt.ylabel("Memory Usage (KB)")
plt.title("Actual vs Predicted Memory Usage")
plt.legend()
plt.grid(True)
plt.tight_layout()

memory_file = OUTPUT_DIR / "memory_actual_vs_predicted.png"

plt.savefig(memory_file, dpi=300)
plt.close()

print("\nMemory graph saved to:")
print(memory_file)


# ============================================
# 6. NETWORK ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(12, 6))

plt.plot(
    plot_df["Actual_Network_KBps"],
    label="Actual Network"
)

plt.plot(
    plot_df["Predicted_Network_KBps"],
    label="Predicted Network"
)

plt.xlabel("Test Sample")
plt.ylabel("Network Throughput (KB/s)")
plt.title("Actual vs Predicted Network Throughput")
plt.legend()
plt.grid(True)
plt.tight_layout()

network_file = OUTPUT_DIR / "network_actual_vs_predicted.png"

plt.savefig(network_file, dpi=300)
plt.close()

print("\nNetwork graph saved to:")
print(network_file)


# ============================================
# 7. COMPLETION MESSAGE
# ============================================

print("\n========================================")
print("PREDICTION GRAPHS CREATED SUCCESSFULLY")
print("========================================")

print("\nGenerated files:")

print("1.", cpu_file)
print("2.", memory_file)
print("3.", network_file)