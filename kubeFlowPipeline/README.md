# KubeFlow Pipelines — MLOps Training Project

A collection of Kubeflow Pipelines (KFP v2) examples demonstrating how to build, compile, and run ML pipelines on Kubernetes — from a basic "Hello World" pipeline to a full Iris classification ML pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture Diagram](#architecture-diagram)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Pipelines](#pipelines)
  - [Hello World Pipeline](#1-hello-world-pipeline)
  - [Iris Classification Pipeline](#2-iris-classification-pipeline)
- [Workflow](#workflow)
- [Compiling the Pipelines](#compiling-the-pipelines)
- [Running Pipelines on Kubeflow](#running-pipelines-on-kubeflow)
- [Security Configuration](#security-configuration)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This project demonstrates end-to-end Kubeflow Pipeline development using the **KFP SDK v2**. It includes:

| Pipeline | Description |
|---|---|
| `hello_pipeline.py` | Simple greeting pipeline — validates KFP setup |
| `iris_pipeline.py` | ML pipeline — loads Iris data, trains a Random Forest classifier, and returns accuracy |

Both pipelines are compiled into portable YAML specs that are uploaded to and executed by the Kubeflow Pipelines backend running on Kubernetes.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Developer Machine                        │
│                                                                 │
│   ┌──────────────┐    compile()    ┌───────────────────────┐   │
│   │  pipeline.py │ ─────────────►  │  pipeline_spec.yaml   │   │
│   │  (KFP SDK v2)│                 │  (IR YAML / Argo spec)│   │
│   └──────────────┘                 └───────────┬───────────┘   │
└───────────────────────────────────────────────┼────────────────┘
                                                 │ Upload via UI / CLI
                                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                           │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Kubeflow Pipelines (KFP)                 │  │
│  │                                                           │  │
│  │   ┌─────────────┐      ┌──────────────────────────────┐  │  │
│  │   │  KFP UI /   │      │     Pipeline Execution       │  │  │
│  │   │  REST API   │─────►│                              │  │  │
│  │   └─────────────┘      │  Step 1: load_data (Pod)     │  │  │
│  │                        │      │                        │  │  │
│  │                        │      ▼                        │  │  │
│  │                        │  Step 2: train_model (Pod)   │  │  │
│  │                        │      │                        │  │  │
│  │                        │      ▼                        │  │  │
│  │                        │  Output: accuracy (float)    │  │  │
│  │                        └──────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Each step runs as an isolated container with:                  │
│    - run_as_user: 1000  (non-root enforcement)                  │
│    - run_as_non_root: true                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Iris Pipeline DAG

```
iris_pipeline
     │
     ▼
┌────────────┐         outputs: features (List[List[float]])
│ load_data  │ ──────────────────────────────────────┐
│            │         outputs: labels (List[int])   │
└────────────┘                                       │
                                                     ▼
                                           ┌─────────────────┐
                                           │  train_model    │
                                           │                 │
                                           │  RandomForest   │
                                           │  Classifier     │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           accuracy: float (printed)
```

---

## Project Structure

```
kubeFlowPipeline/
├── hello_pipeline.py          # Hello World pipeline definition (KFP component + pipeline)
├── hello_world_pipeline.yaml  # Compiled IR YAML for hello_pipeline
├── iris_pipeline.py           # Iris ML pipeline (load data → train model)
├── iris_pipeline.yaml         # Compiled IR YAML for iris_pipeline
└── README.md                  # This file
```

---

## Prerequisites

### Local Machine

| Requirement | Version | Notes |
|---|---|---|
| Python | >= 3.8 | 3.10 recommended |
| pip | latest | `pip install --upgrade pip` |
| kfp SDK | 2.16.0 | Kubeflow Pipelines SDK v2 |
| kfp-kubernetes | latest | For security context support |

### Kubernetes Cluster

| Requirement | Notes |
|---|---|
| Kubernetes | >= 1.24 |
| Kubeflow Pipelines | >= 2.x installed on cluster |
| `kubectl` | Configured to point at your cluster |
| Container registry access | For pulling `python:3.10-slim`, `python:3.11` images |

---

## Installation Guide

### 1. Clone / Navigate to the Project

```bash
cd /home/amol/kra-assignments/mlops_assignments/Agentic_AI_and_MLOps_Training/kubeFlowPipeline
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install kfp==2.16.0
pip install kfp-kubernetes
```

> **Note:** `kfp-kubernetes` is required for `kubernetes.set_security_context()` used in both pipelines.

### 4. Verify Installation

```bash
python -c "import kfp; print(kfp.__version__)"
# Expected: 2.16.0

python -c "from kfp import kubernetes; print('kfp-kubernetes OK')"
```

### 5. (Optional) Install ML Dependencies for Local Testing

```bash
pip install pandas scikit-learn
```

---

## Pipelines

### 1. Hello World Pipeline

**File:** [hello_pipeline.py](hello_pipeline.py)  
**Compiled YAML:** [hello_world_pipeline.yaml](hello_world_pipeline.yaml)

#### Description

A minimal pipeline with a single `say_hello` component. Validates that KFP SDK, compilation, and Kubernetes security context configuration are all working correctly.

#### Component

| Component | Input | Output | Base Image |
|---|---|---|---|
| `say_hello` | `name: str` | `str` (greeting message) | `python:3.11` |

#### Pipeline Parameters

| Parameter | Default | Description |
|---|---|---|
| `recipient` | `"World"` | Name to greet |

#### Example Output

```
Hello, World!
```

---

### 2. Iris Classification Pipeline

**File:** [iris_pipeline.py](iris_pipeline.py)  
**Compiled YAML:** [iris_pipeline.yaml](iris_pipeline.yaml)

#### Description

An end-to-end ML pipeline that:
1. Loads the Iris dataset (150 samples, 4 features, 3 classes)
2. Trains a Random Forest Classifier
3. Evaluates and prints model accuracy

Data is passed between steps as Python lists (no artifact/file I/O) — making it suitable for lightweight in-memory workflows.

#### Components

| Component | Inputs | Outputs | Base Image | Additional Packages |
|---|---|---|---|---|
| `load_data` | — | `features: List[List[float]]`, `labels: List[int]` | `python:3.10-slim` | `pandas`, `scikit-learn` |
| `train_model` | `features`, `labels` | `accuracy: float` | `python:3.10-slim` | `scikit-learn` |

#### Model Details

| Property | Value |
|---|---|
| Dataset | Iris (sklearn built-in) |
| Train/Test Split | 80% / 20% |
| Algorithm | `RandomForestClassifier` (default params) |
| Metric | Accuracy Score |

---

## Workflow

```
1. Write Pipeline
      │
      │  Define @dsl.component functions
      │  Define @dsl.pipeline orchestration
      │  Apply kubernetes.set_security_context()
      │
      ▼
2. Compile Pipeline
      │
      │  compiler.Compiler().compile(pipeline_func, 'output.yaml')
      │
      ▼
3. Upload to Kubeflow UI / CLI
      │
      │  Open KFP UI → Pipelines → Upload Pipeline → select .yaml
      │  OR use kfp.Client() to upload programmatically
      │
      ▼
4. Create a Run
      │
      │  Select pipeline version → Create Run
      │  Provide input parameters (e.g., recipient="Alice")
      │
      ▼
5. Monitor Execution
      │
      │  View DAG graph in KFP UI
      │  Click each node to see logs, inputs, outputs
      │
      ▼
6. Inspect Results
         View output parameters / metrics in the Run details panel
```

---

## Compiling the Pipelines

Run the pipeline Python files directly to regenerate the YAML specs:

```bash
# Compile Hello World pipeline
python hello_pipeline.py
# Output: hello_world_pipeline.yaml

# Compile Iris ML pipeline
python iris_pipeline.py
# Output: iris_pipeline.yaml
```

Both scripts call `compiler.Compiler().compile(...)` in their `if __name__ == "__main__":` block.

---

## Running Pipelines on Kubeflow

### Option A — Kubeflow UI (Manual)

1. Open the Kubeflow Pipelines dashboard (e.g., `http://<kubeflow-host>/pipeline`)
2. Go to **Pipelines** → **Upload Pipeline**
3. Upload `hello_world_pipeline.yaml` or `iris_pipeline.yaml`
4. Go to **Runs** → **Create Run**
5. Select the uploaded pipeline, set parameters, and submit

### Option B — KFP Python Client (Programmatic)

```python
import kfp

client = kfp.Client(host="http://<kubeflow-host>/pipeline")

# Upload and run the hello world pipeline
run = client.create_run_from_pipeline_package(
    pipeline_file="hello_world_pipeline.yaml",
    arguments={"recipient": "Alice"},
    run_name="hello-run-01",
)
print(run.run_id)
```

```python
# Upload and run the Iris pipeline
run = client.create_run_from_pipeline_package(
    pipeline_file="iris_pipeline.yaml",
    run_name="iris-run-01",
)
print(run.run_id)
```

### Option C — kubectl (Advanced)

```bash
kubectl apply -f iris_pipeline.yaml -n kubeflow
```

---

## Security Configuration

Both pipelines apply Kubernetes Pod security context to each step using `kfp-kubernetes`:

```python
from kfp import kubernetes

kubernetes.set_security_context(
    task,
    run_as_user=1000,      # Run as UID 1000 (non-root)
    run_as_non_root=True,  # Enforce non-root at runtime
)
```

This satisfies `runAsNonRoot` Pod Security Admission (PSA) policies enforced on the cluster.

| Setting | Value | Purpose |
|---|---|---|
| `run_as_user` | `1000` | Runs container process as UID 1000 |
| `run_as_non_root` | `True` | Kubernetes enforces process is not root |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'kfp'`
```bash
pip install kfp==2.16.0
```

### `ModuleNotFoundError: No module named 'kfp.kubernetes'`
```bash
pip install kfp-kubernetes
```

### Pipeline pod fails with `runAsNonRoot` error
Ensure `kubernetes.set_security_context(task, run_as_user=1000, run_as_non_root=True)` is applied to all tasks, which is already done in both pipelines.

### YAML not updating after code changes
Re-run the compile step:
```bash
python hello_pipeline.py
python iris_pipeline.py
```

### KFP Client connection refused
Verify Kubeflow is running and port-forward if needed:
```bash
kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow
# Then access: http://localhost:8080
```

---

## Dependencies Summary

| Package | Version | Used In |
|---|---|---|
| `kfp` | 2.16.0 | Both pipelines (SDK, compiler, DSL) |
| `kfp-kubernetes` | latest | Both pipelines (security context) |
| `pandas` | latest | `iris_pipeline` — `load_data` component |
| `scikit-learn` | latest | `iris_pipeline` — both components |

> `pandas` and `scikit-learn` are installed **inside the pipeline containers at runtime** via `packages_to_install` — they do not need to be installed locally unless you run components directly.
