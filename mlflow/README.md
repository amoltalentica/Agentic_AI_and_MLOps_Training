# Kidney Disease Classification — MLflow + DVC + Minikube

A production-grade **CNN-based kidney CT scan classifier** that distinguishes between **Normal** and **Tumor** kidney images. Built on VGG16 (transfer learning), tracked with MLflow, orchestrated with DVC, containerised with Docker, and deployed on **Minikube (local Kubernetes)** — no cloud dependencies.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Minikube Cluster                              │
│   Namespace: cnn-classifier                                          │
│                                                                      │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐   │
│  │   cnn-classifier Pod     │    │    mlflow-server Pod          │   │
│  │   (Flask + VGG16)        │    │    (MLflow Tracking UI)       │   │
│  │   Port: 8080             │◄──►│    Port: 5000                 │   │
│  │   Image: cnn-classifier  │    │    Image: mlflow/mlflow:2.2.2 │   │
│  └──────────┬───────────────┘    └──────────────┬───────────────┘   │
│             │ PVC (artifacts/)                  │ PVC (mlruns/)      │
│             │ 5Gi                               │ 3Gi                │
│  ┌──────────▼──────────────────────────────────▼───────────────┐   │
│  │              PersistentVolumeClaims (hostPath)                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                                │
    localhost:8080                   localhost:5000
    (kubectl port-forward)           (kubectl port-forward)
         │                                │
    Flask Web UI /                   MLflow Tracking UI
    POST /predict                    Experiments & Models
    POST /train
```

### ML Pipeline (DVC DAG)

```
[Data Ingestion]
      │  Downloads CT kidney dataset (Google Drive → artifacts/data_ingestion/)
      ▼
[Prepare Base Model]
      │  VGG16 (ImageNet weights, top removed) + Dense(2, softmax) head
      ▼
[Model Training]
      │  Fine-tune classification head (10 epochs, batch 16, SGD lr=0.01)
      │  Augmentation: rotation, flip, shift, shear, zoom
      ▼
[Model Evaluation + MLflow Logging]
      │  Logs: loss, accuracy → MLflow server at http://mlflow-server:5000
      │  Registers model as: VGG16Model
      ▼
[Trained Model → artifacts/training/model.h5]
```

---

## Project Structure

```
mlflow/
├── app.py                          # Flask application (endpoints: /, /train, /predict)
├── main.py                         # Runs full DVC pipeline (all 4 stages)
├── Dockerfile                      # tensorflow/tensorflow:2.12.0 base image
├── params.yaml                     # Training hyperparameters
├── config/config.yaml              # Paths, URLs, artifact roots
├── dvc.yaml                        # DVC pipeline stages
├── setup.py                        # Package: cnnClassifier (src/)
├── requirements.txt
├── model/model.h5                  # Fallback model (used if no trained model on PVC)
├── sample_images/                  # Real kidney CT scan test images
│   ├── normal_01.jpg  normal_02.jpg  normal_03.jpg
│   └── tumor_01.jpg   tumor_02.jpg   tumor_03.jpg
├── src/cnnClassifier/
│   ├── components/
│   │   ├── data_ingestion.py       # Downloads and extracts dataset
│   │   ├── prepare_base_model.py   # Builds VGG16 + classification head
│   │   ├── model_training.py       # Trains with ImageDataGenerator
│   │   └── model_evaluation_mlflow.py  # Evaluates & logs to MLflow
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_training.py
│   │   ├── stage_04_model_evaluation.py
│   │   └── prediction.py           # PredictionPipeline (used by Flask /predict)
│   ├── config/configuration.py     # ConfigurationManager (reads config.yaml + params.yaml)
│   ├── entity/config_entity.py     # Dataclass configs for each stage
│   └── utils/common.py             # Helpers: read_yaml, decodeImage, encodeImage
├── k8s/
│   ├── namespace.yaml              # Namespace: cnn-classifier
│   ├── secret.yaml                 # MLFLOW_TRACKING_URI env var
│   ├── pvc.yaml                    # 5Gi PVC for app artifacts/
│   ├── deployment.yaml             # CNN classifier Deployment (imagePullPolicy: Never)
│   ├── service.yaml                # NodePort 8080 → 30080
│   ├── mlflow-pvc.yaml             # 3Gi PVC for MLflow mlruns/
│   ├── mlflow-deployment.yaml      # MLflow server Deployment + Service (5000 → 30500)
│   ├── deploy.sh                   # Full deploy script (builds image + applies manifests)
│   └── tunnel.sh                   # WSL2 port-forward helper
├── templates/index.html            # Web UI (Upload + Predict)
└── test_predict.py                 # CLI test script for /predict endpoint
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | VGG16 (TensorFlow/Keras 2.12) |
| Experiment Tracking | MLflow 2.2.2 |
| Pipeline Orchestration | DVC |
| Web Framework | Flask + Flask-CORS |
| Containerisation | Docker (`tensorflow/tensorflow:2.12.0` base) |
| Orchestration | Kubernetes (Minikube v1.38.1, Docker driver) |
| Storage | Kubernetes PersistentVolumeClaims (hostPath) |
| Dataset | CT KIDNEY DATASET (Normal + Tumor classes) |

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Docker | 20+ | `sudo apt install docker.io` |
| Minikube | 1.30+ | [minikube.sigs.k8s.io](https://minikube.sigs.k8s.io/docs/start/) |
| kubectl | 1.27+ | `sudo snap install kubectl --classic` |
| Python | 3.8+ | System Python or conda |

---

## Installation & Deployment Guide

### Option A — Full Kubernetes Deployment (Minikube)

#### Step 1 — Clone the repository

```bash
git clone https://github.com/krishnaik06/Kidney-Disease-Classification-Deep-Learning-Project
cd mlflow
```

#### Step 2 — Run the deploy script

```bash
bash k8s/deploy.sh
```

This script automatically:
1. Checks if Minikube is healthy (recreates cluster if broken)
2. Starts Minikube with 4 CPUs and 6GB RAM
3. Points Docker to Minikube's daemon (`eval $(minikube docker-env)`)
4. Builds the `cnn-classifier:latest` Docker image inside Minikube
5. Applies all Kubernetes manifests in order:
   - Namespace → Secrets → PVCs → MLflow server → App deployment → Services
6. Waits for MLflow server to be ready before deploying the app

#### Step 3 — Open port-forward tunnels (WSL2 / remote access)

```bash
bash k8s/tunnel.sh
```

Or manually:
```bash
kubectl port-forward -n cnn-classifier svc/cnn-classifier-service 8080:8080 --address=0.0.0.0 &
kubectl port-forward -n cnn-classifier svc/mlflow-server 5000:5000 --address=0.0.0.0 &
```

#### Step 4 — Access the services

| Service | URL |
|---|---|
| Flask Web UI | http://localhost:8080 |
| MLflow Tracking UI | http://localhost:5000 |

---

### Option B — Local Development (no Kubernetes)

#### Step 1 — Create environment

```bash
conda create -n cnncls python=3.8 -y
conda activate cnncls
pip install -r requirements.txt
pip install -e .
```

#### Step 2 — Start a local MLflow server

```bash
mlflow server --host 0.0.0.0 --port 5000 &
export MLFLOW_TRACKING_URI=http://localhost:5000
```

#### Step 3 — Run the Flask app

```bash
python app.py
```

App will be available at http://localhost:8080

---

## Running the Training Pipeline

### Via Web UI

Navigate to http://localhost:8080 and click **Train** or:

```bash
curl -X POST http://localhost:8080/train
```

### Via DVC (local dev)

```bash
dvc repro          # runs all 4 stages (skips up-to-date stages)
dvc dag            # visualise the pipeline DAG
```

### Via main.py directly

```bash
python main.py
```

The full pipeline runs these stages in order:

1. **Data Ingestion** — downloads `data.zip` from Google Drive, extracts to `artifacts/data_ingestion/kidney-ct-scan-image/`
2. **Prepare Base Model** — downloads VGG16 weights (ImageNet), adds Dense(2, softmax) head, saves to `artifacts/prepare_base_model/`
3. **Model Training** — trains for 10 epochs with augmentation, saves to `artifacts/training/model.h5`
4. **Model Evaluation** — evaluates on validation set, logs `loss` + `accuracy` to MLflow, registers model as `VGG16Model`

---

## Making Predictions

### Option 1 — Web UI

1. Open http://localhost:8080
2. Click **Upload** → select a kidney CT scan image (JPEG/PNG)
3. Click **Predict** → result appears in the **Prediction Results** panel

### Option 2 — test_predict.py (CLI)

```bash
# Test with a real Normal kidney CT scan
python3 test_predict.py --image sample_images/normal_01.jpg

# Test with a real Tumor kidney CT scan
python3 test_predict.py --image sample_images/tumor_01.jpg
```

### Option 3 — Python one-liner

```bash
python3 -c "
import base64, json, urllib.request
with open('sample_images/tumor_01.jpg','rb') as f:
    b64 = base64.b64encode(f.read()).decode()
req = urllib.request.Request('http://localhost:8080/predict',
    data=json.dumps({'image': b64}).encode(),
    headers={'Content-Type':'application/json'})
print(json.loads(urllib.request.urlopen(req).read()))
"
```

### Sample Test Images

Real CT scans from the training dataset are included in `sample_images/`:

| File | Class | Expected Result |
|---|---|---|
| `normal_01.jpg` | Normal kidney | `{"image": "Normal"}` |
| `normal_02.jpg` | Normal kidney | `{"image": "Normal"}` |
| `normal_03.jpg` | Normal kidney | `{"image": "Normal"}` |
| `tumor_01.jpg` | Kidney tumor | `{"image": "Tumor"}` |
| `tumor_02.jpg` | Kidney tumor | `{"image": "Tumor"}` |
| `tumor_03.jpg` | Kidney tumor | `{"image": "Tumor"}` |

> **Note:** Use the real CT scan images (`*_01/02/03.jpg`). Synthetic/drawn images will not produce reliable predictions since the model was trained on real medical scans.

---

## Model Configuration

All hyperparameters are in [`params.yaml`](params.yaml):

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]   # VGG16 input size
BATCH_SIZE: 16
INCLUDE_TOP: False           # Remove VGG16 classification head
EPOCHS: 10
CLASSES: 2                   # Normal, Tumor
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

To change epochs without rebuilding the image:
```bash
# Edit params.yaml, then patch the running pod
kubectl -n cnn-classifier exec deploy/cnn-classifier -- \
  sed -i 's/EPOCHS: 10/EPOCHS: 20/' /app/params.yaml

# Trigger retraining
curl -X POST http://localhost:8080/train
```

---

## MLflow Experiment Tracking

After training, open http://localhost:5000 to view:

- **Experiments** — each `/train` call creates a new MLflow run
- **Metrics** — `loss` and `accuracy` logged per run
- **Model Registry** — trained model registered as `VGG16Model` (versioned)
- **Artifacts** — saved model artifacts linked to each run

MLflow tracking URI is configured via environment variable (set in `k8s/secret.yaml`):
```
MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

---

## Kubernetes Resources

| Resource | Name | Details |
|---|---|---|
| Namespace | `cnn-classifier` | Isolates all project resources |
| Deployment | `cnn-classifier` | Flask app, 1 replica, 2Gi/4Gi mem |
| Deployment | `mlflow-server` | MLflow tracking server, 1 replica |
| Service | `cnn-classifier-service` | NodePort 8080 → 30080 |
| Service | `mlflow-server` | NodePort 5000 → 30500 |
| PVC | `artifacts-pvc` | 5Gi — app artifacts, model, logs |
| PVC | `mlflow-pvc` | 3Gi — MLflow mlruns and artifacts |
| Secret | `cnn-classifier-secret` | MLFLOW_TRACKING_URI |

### Useful kubectl commands

```bash
# Check pod status
kubectl get pods -n cnn-classifier

# View app logs (training progress)
kubectl logs -f -n cnn-classifier deploy/cnn-classifier

# View MLflow server logs
kubectl logs -f -n cnn-classifier deploy/mlflow-server

# Shell into the app container
kubectl exec -it -n cnn-classifier deploy/cnn-classifier -- bash

# Check persistent volumes
kubectl get pvc -n cnn-classifier

# Restart the app pod (e.g. after model update)
kubectl rollout restart deployment/cnn-classifier -n cnn-classifier
```

---

## API Reference

### `GET /`
Returns the web UI (`templates/index.html`).

### `POST /train`
Triggers the full 4-stage ML pipeline (`main.py`). Runs asynchronously inside the container. Check progress via pod logs.

```bash
curl -X POST http://localhost:8080/train
```

### `POST /predict`
Classifies a kidney CT scan image.

**Request body:**
```json
{ "image": "<base64-encoded JPEG string>" }
```

**Response:**
```json
[{ "image": "Normal" }]
# or
[{ "image": "Tumor" }]
```

---

## Development Workflow

```
1. Edit source code (src/cnnClassifier/...)
2. Update config.yaml / params.yaml if needed
3. Run locally:        python main.py
4. Or run via DVC:     dvc repro
5. Rebuild image:      eval $(minikube docker-env) && docker build -t cnn-classifier:latest .
6. Redeploy:           kubectl rollout restart deployment/cnn-classifier -n cnn-classifier
```

---

## DVC Pipeline

```bash
dvc init           # initialise DVC (already done)
dvc repro          # reproduce pipeline (only runs changed stages)
dvc dag            # print the pipeline DAG
```

Pipeline stages defined in [`dvc.yaml`](dvc.yaml):

| Stage | Input | Output |
|---|---|---|
| `data_ingestion` | `config.yaml` | `artifacts/data_ingestion/` |
| `prepare_base_model` | `params.yaml` | `artifacts/prepare_base_model/` |
| `training` | base model + data | `artifacts/training/model.h5` |
| `evaluation` | trained model + data | `scores.json`, MLflow run |

---

## Troubleshooting

### Port-forward drops / services unreachable

```bash
# Restart both tunnels
pkill -f "kubectl port-forward"
kubectl port-forward -n cnn-classifier svc/cnn-classifier-service 8080:8080 --address=0.0.0.0 &
kubectl port-forward -n cnn-classifier svc/mlflow-server 5000:5000 --address=0.0.0.0 &
```

### Pod in CrashLoopBackOff

```bash
kubectl logs -n cnn-classifier deploy/cnn-classifier
```

Common causes:
- `ModuleNotFoundError: No module named 'cnnClassifier'` → rebuild image (`bash k8s/deploy.sh`)
- MLflow connection refused → check MLflow pod is running (`kubectl get pods -n cnn-classifier`)

### Minikube cluster broken / API unresponsive

```bash
minikube delete && bash k8s/deploy.sh
```

### Training returns all "Normal" (model not converged)

- Ensure `EPOCHS: 10` in `params.yaml` (not `EPOCHS: 1`)
- Trigger retraining: `curl -X POST http://localhost:8080/train`
- Check progress: `kubectl logs -f -n cnn-classifier deploy/cnn-classifier`

### Predictions always return "Normal" after pod restart

The trained model persists on PVC at `artifacts/training/model.h5`. The prediction pipeline automatically uses this over the image's `model/model.h5` fallback. If missing, run `/train` once to regenerate it.

---

## About MLflow & DVC

**MLflow:**
- Production-grade experiment tracking
- Logs metrics, parameters, and artifacts per run
- Model registry with versioning (`VGG16Model`)
- UI at http://localhost:5000

**DVC (Data Version Control):**
- Lightweight pipeline orchestration
- Tracks data and model files in Git-compatible way
- Only reruns stages whose inputs have changed
- `dvc dag` renders the full pipeline graph

---

## Adding a New Pipeline Stage (Developer Guide)

When extending the project with a new pipeline stage, follow this order:

1. Update `config/config.yaml` — add artifact paths for the new stage
2. Update `params.yaml` — add any new hyperparameters
3. Update `src/cnnClassifier/entity/config_entity.py` — add a new dataclass
4. Update `src/cnnClassifier/config/configuration.py` — add a getter method
5. Implement in `src/cnnClassifier/components/` — core logic
6. Add a pipeline stage in `src/cnnClassifier/pipeline/`
7. Wire into `main.py`
8. Add a stage entry to `dvc.yaml`


---

