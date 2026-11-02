# Adaptive Multi-Resource Deep Learning-Based Predictive Container Auto-Scaling Framework for Kubernetes

## Overview

This project presents an **Adaptive Multi-Resource Deep Learning-Based Predictive Container Auto-Scaling Framework for Kubernetes**.

The main objective is to predict future resource requirements of containerized applications using historical workload data and make proactive scaling decisions.

Traditional reactive autoscaling responds after resource utilization increases. Our approach uses a Long Short-Term Memory (LSTM) deep learning model to predict three important resource metrics:

- CPU Usage
- Memory Usage
- Network Throughput

The predicted resource values are processed by a weighted scaling controller to generate scaling decisions such as **SCALE UP, KEEP, and SCALE DOWN**.

The framework integrates Docker, Kubernetes, Minikube, Prometheus, and Grafana to demonstrate a predictive autoscaling prototype.

---

## Problem Statement

Containerized applications experience changing workloads, which can lead to fluctuating CPU, memory, and network requirements.

Traditional reactive autoscaling generally makes decisions based on current resource utilization. This may introduce a delay between an increase in workload and the corresponding scaling action.

Additionally, relying only on CPU utilization may overlook memory and network resource requirements.

This project addresses these challenges by predicting multiple resource metrics using an LSTM model and using the predictions to support adaptive Kubernetes container scaling decisions.

---

## Project Objectives

1. Study existing reactive and predictive container autoscaling techniques.
2. Prepare historical virtual machine workload data for deep learning.
3. Develop an LSTM model to predict CPU, memory, and network usage.
4. Implement a model fine-tuning mechanism to examine prediction improvements.
5. Develop a weighted multi-resource scaling decision mechanism.
6. Integrate the predictive controller with Kubernetes.
7. Use Prometheus and Grafana for resource monitoring and visualization.
8. Evaluate prediction performance and compare predictive and reactive CPU-threshold decisions.

---

## System Architecture

The framework consists of two connected workflows:

### 1. Offline Model Development

Historical VM workload data is processed and used to train the LSTM model.

**Workflow:**

Bitbrains VM Workload Dataset → Data Preprocessing → Chronological Data Splitting → Normalization → Sequence Generation → LSTM Training → Model Evaluation → Fine-Tuning

### 2. Live Predictive Autoscaling Prototype

The trained model is integrated with a Kubernetes-based monitoring and scaling workflow.

**Workflow:**

Containerized Application → Kubernetes / Minikube → Prometheus Monitoring → LSTM Prediction → Weighted Scaling Controller → Kubernetes Replica Scaling

Grafana is used to visualize the collected resource metrics.

---

## Dataset

We use the **GWA-T-12 Bitbrains fastStorage VM Workload Trace Dataset** for historical resource prediction.

The dataset contains recorded virtual machine workload information.

### Dataset Summary

| Description | Details |
|---|---|
| Dataset | GWA-T-12 Bitbrains fastStorage |
| CSV Files Used | 10 |
| Cleaned Dataset Rows | 146,393 |
| Resource Features | 3 |
| Sequence Length | 12 observations |
| Test Sequences | 21,844 |

### Selected Features

| Feature | Description |
|---|---|
| CPU Usage (%) | CPU utilization of the virtual machine |
| Memory Usage (KB) | Memory consumed by the virtual machine |
| Network Throughput (KB/s) | Combined received and transmitted network throughput |

Network throughput is calculated by adding received and transmitted network throughput.

### Why This Dataset?

- Provides historical virtual machine workload observations.
- Supports time-series resource prediction.
- Contains CPU, memory, and network-related measurements.
- Allows chronological training, validation, and testing.
- Provides a workload trace for evaluating resource prediction models.

---

## Data Preprocessing

The raw workload data is prepared before being provided to the LSTM model.

The preprocessing pipeline performs the following operations:

1. Loads the selected CSV workload files.
2. Cleans column names and data values.
3. Converts required columns into numeric values.
4. Calculates total network throughput.
5. Assigns VM identifiers.
6. Removes rows with missing required values.
7. Combines the cleaned data into a single dataset.
8. Sorts the observations by VM identifier and timestamp.

The cleaned dataset is saved as:

`dataset/clean_dataset.csv`

### Data Splitting

Each VM's observations are divided chronologically:

| Dataset | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Train the LSTM model |
| Validation | 15% | Monitor and tune the model |
| Testing | 15% | Evaluate the trained model |

### Normalization

We use **MinMaxScaler** to normalize CPU, memory, and network features.

The scaler is fitted only on the training data and then applied to the corresponding validation and test data.

A separate scaler is maintained for each VM.

### Sequence Generation

The model uses a sliding-window approach.

- Input: Previous 12 observations.
- Features per observation: CPU, memory, and network.
- Output: Next observation's three resource values.

Therefore, the input shape for one sequence is:

`(12, 3)`

The sequence generation process is performed independently within the training, validation, and test splits.

---

## LSTM Model Architecture

We use a multi-output Long Short-Term Memory (LSTM) neural network to learn temporal patterns in historical resource usage.

### Model Configuration

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 12 time steps, 3 features | Receives historical resource observations |
| LSTM | 64 units | Learns temporal patterns |
| Dropout | 0.2 | Helps reduce overfitting |
| Dense | 32 neurons, ReLU | Processes learned features |
| Output | 3 neurons, Linear | Predicts CPU, memory, and network usage |

**Total trainable parameters:** 58,763

### Model Output

For each input sequence, the model predicts:

1. Next CPU usage value.
2. Next memory usage value.
3. Next network throughput value.

### Training Configuration

- Deep Learning Framework: TensorFlow / Keras
- Model: LSTM
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Early Stopping: Used during training
- Model Checkpointing: Used to save the best model

---

## Adaptive Fine-Tuning

After training the baseline LSTM model, we perform an additional fine-tuning experiment.

The saved baseline model is trained for five additional epochs using a selected portion of the validation data.

### Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| Initial Model | Baseline LSTM |
| Additional Epochs | 5 |
| Learning Rate | 0.0001 |
| Fine-Tuning Data | Selected final 20% of concatenated validation arrays |
| Shuffling | Disabled |

The objective is to examine whether additional training on the selected validation window improves prediction performance.

The updated model is saved separately as:

`adaptive_lstm.keras`

**Note:** This implementation demonstrates validation-window fine-tuning. It is not continuous online learning from live production workloads.

---

## Multi-Resource Scaling Decision Mechanism

The scaling controller uses predicted CPU, memory, and network values to calculate a weighted resource-pressure score.

### Resource Weights

| Resource | Weight |
|---|---:|
| CPU | 0.50 |
| Memory | 0.30 |
| Network | 0.20 |

### Weighted Pressure Formula

`Pressure = (0.50 × CPU) + (0.30 × Memory) + (0.20 × Network)`

The resource predictions are normalized and processed by the controller before calculating the weighted score.

### Scaling Thresholds

| Condition | Scaling Decision |
|---|---|
| Pressure ≥ 0.65 | SCALE UP |
| Pressure ≤ 0.35 | SCALE DOWN |
| 0.35 < Pressure < 0.65 | KEEP |

### Scaling Safeguards

The live controller includes:

- Minimum replica count: 1
- Maximum replica count: 5
- Three consecutive matching decisions before scaling
- 300-second cooldown between scaling actions

These safeguards help prevent unnecessary or rapid replica changes.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset preprocessing |
| NumPy | Numerical operations and sequence preparation |
| Scikit-learn | MinMaxScaler and evaluation utilities |
| TensorFlow / Keras | LSTM model development and training |
| Joblib | Saving and loading VM-specific scalers |
| Docker | Application containerization |
| Kubernetes | Container orchestration and replica management |
| Minikube | Local Kubernetes environment |
| Prometheus | Resource metrics collection |
| Grafana | Metrics visualization |
| Git and GitHub | Version control and project hosting |
| VS Code | Development environment |

---

## Experimental Results

### 1. Baseline LSTM Evaluation

The baseline LSTM model is evaluated using **21,844 held-out test sequences**.

The following results represent prediction errors in the original resource units.

| Resource | MAE | RMSE |
|---|---:|---:|
| CPU | 0.278457 percentage points | 2.698687 percentage points |
| Memory | 48,832.526582 KB | 465,576.861255 KB |
| Network | 11.400737 KB/s | 93.652781 KB/s |

**MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.

**RMSE (Root Mean Squared Error):** Gives more weight to larger prediction errors.

Lower values indicate smaller prediction errors.

### 2. Baseline vs Fine-Tuned Model

The following comparison uses normalized evaluation metrics.

| Metric | Baseline LSTM | Fine-Tuned LSTM |
|---|---:|---:|
| Overall MAE | 0.006441 | 0.005036 |
| Overall RMSE | 0.169601 | 0.169636 |

**Observations:**

- Overall MAE decreased by approximately 21.82%.
- Overall RMSE increased slightly by approximately 0.02%.
- Fine-tuning produced mixed results rather than improving every metric.

### 3. Offline CPU-Threshold Decision Comparison

We performed a separate offline experiment to compare reactive and predictive CPU-threshold decisions.

The threshold used in this experiment is **65% CPU utilization**.

- Reactive approach: Uses actual CPU usage.
- Predictive approach: Uses LSTM-predicted CPU usage.

If CPU usage is at least 65%, the corresponding rule produces SCALE UP; otherwise, it produces KEEP.

#### Decision Comparison

| Reactive Label | Predictive KEEP | Predictive SCALE UP |
|---|---:|---:|
| Reactive KEEP | 21,737 | 4 |
| Reactive SCALE UP | 37 | 66 |

### Key Observations

- Total test samples: 21,844
- Matching decisions: 21,803
- Overall agreement: 99.81%
- Reactive SCALE UP labels: 103
- Predictive SCALE UP labels: 70
- Reactive SCALE UP samples also identified by predictive rule: 66
- Reactive SCALE UP samples labelled KEEP by predictive rule: 37

**Important Interpretation:**

The 99.81% value represents overall decision agreement, not LSTM prediction accuracy.

The agreement is dominated by the large KEEP class. This offline CPU-only comparison does not prove improved live autoscaling, lower cloud cost, better resource utilization, or preserved application service quality.

The complete project also includes a separate multi-resource scaling controller that uses CPU, memory, and network predictions.

---

## Kubernetes Integration

The project includes a local Kubernetes prototype.

### Implementation Workflow

1. The application is packaged using Docker.
2. The containerized application is deployed in Minikube.
3. Kubernetes manages the application Pods and Deployment.
4. Prometheus collects resource metrics.
5. The Python controller queries Prometheus for recent metrics.
6. The LSTM model generates resource predictions.
7. The weighted scaling controller determines the scaling action.
8. After the required confirmation and cooldown conditions, the controller can execute a Kubernetes replica-scaling command.
9. Grafana visualizes the monitoring metrics.

The live prototype demonstrates the integration of prediction, monitoring, and Kubernetes replica management.

It should not be interpreted as a production-scale performance benchmark.

---

## Project Structure

The following is a representative organization of the main project modules:

```text
predictive-container-autoscaling/
│
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── clean_dataset.csv
│
├── preprocessing/
│   ├── preprocess.py
│   └── prepare_sequences.py
│
├── model/
│   ├── train_lstm.py
│   ├── evaluate_lstm.py
│   ├── adaptive_update.py
│   └── compare_models.py
│
├── scaling/
│   ├── scaling_decision.py
│   └── auto_scaler.py
│
├── results/
│
├── Kubernetes configuration files
│
└── README.md
```

The exact directory and configuration-file names may differ in the final repository.

---

## Running the Project

### Prerequisites

Install or configure the following:

- Python
- Required Python libraries
- Docker
- Kubernetes / Minikube
- Prometheus
- Grafana
- kubectl

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/predictive-container-autoscaling.git
```

```bash
cd predictive-container-autoscaling
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the dependencies required by the project:

```bash
pip install pandas numpy scikit-learn tensorflow joblib matplotlib
```

Install any additional dependencies required by the application and monitoring controller.

### Step 4: Prepare the Dataset

Place the selected raw workload CSV files in:

```text
dataset/raw/
```

Run:

```bash
python preprocessing/preprocess.py
```

Then generate the training, validation, and test sequences:

```bash
python preprocessing/prepare_sequences.py
```

### Step 5: Train the LSTM Model

```bash
python model/train_lstm.py
```

### Step 6: Evaluate the Model

```bash
python model/evaluate_lstm.py
```

### Step 7: Run the Fine-Tuning Experiment

```bash
python model/adaptive_update.py
```

### Step 8: Compare the Models

```bash
python model/compare_models.py
```

### Step 9: Configure Kubernetes and Monitoring

Configure Docker, Minikube, Kubernetes Deployment and Service resources, Prometheus, and Grafana according to the configuration files included in the repository.

The live controller requires a running application, accessible Prometheus metrics, the trained model, and the required Kubernetes permissions.

**Note:** These commands describe the main pipeline. Verify the project paths, dependencies, model locations, and configuration settings before execution. Live scaling should only be enabled in a controlled test environment.

---

## Limitations

1. The historical dataset contains duplicate records and irregular timestamp intervals.
2. Twelve observations do not always represent a fixed time duration.
3. The LSTM output layer can generate invalid resource values, such as negative predictions or CPU predictions above 100%.
4. Fine-tuning results are mixed, with MAE improving while RMSE increases slightly.
5. The current fine-tuning mechanism is not continuous online learning.
6. The live controller uses a recent-window scaler, which differs from the VM-specific scalers used during historical training.
7. The reported 99.81% agreement comes from a separate offline CPU-only threshold comparison.
8. More controlled experiments are required to establish live performance, cost savings, and generalization to different workloads.
9. The implementation is a local Kubernetes prototype and requires further validation in production environments.

---

## Future Scope

The project can be extended in the following directions:

- Improve data quality by handling duplicate records and irregular timestamps.
- Use consistent timestamp intervals for time-series forecasting.
- Align live normalization with the historical training pipeline.
- Implement robust continuous online learning and workload-drift adaptation.
- Apply output constraints to prevent physically invalid resource predictions.
- Evaluate multi-resource scaling decisions under controlled workloads.
- Compare the framework with Kubernetes Horizontal Pod Autoscaler (HPA).
- Measure response time, resource utilization, replica changes, and application performance.
- Investigate cloud cost optimization and energy-efficient autoscaling.
- Evaluate the model on additional datasets and real-world workloads.

---

## Conclusion

This project demonstrates a prototype for adaptive predictive container autoscaling using deep learning and Kubernetes.

By predicting CPU, memory, and network usage using an LSTM model, the framework aims to support proactive resource allocation instead of relying exclusively on current resource utilization.

The integration of Docker, Kubernetes, Minikube, Prometheus, and Grafana demonstrates a monitoring-to-scaling workflow.

The experimental results provide preliminary evidence of resource forecasting capability and model fine-tuning behavior. Further improvements and controlled live evaluations are required to establish the framework's effectiveness, reliability, and production readiness.

---

## Project Information

**Project Title:** An Adaptive Multi-Resource Deep Learning-Based Predictive Container Auto-Scaling Framework for Kubernetes

**Domain:** Operating Systems, Cloud Computing, Container Orchestration, Deep Learning

**Primary Model:** Long Short-Term Memory (LSTM)

**Dataset:** GWA-T-12 Bitbrains fastStorage VM Workload Trace

**Repository:** Add the GitHub repository URL here after creating the repository.

---

## Acknowledgment

We acknowledge the Grid Workloads Archive and the Bitbrains workload trace dataset providers for making historical VM workload data available for research and experimentation.


