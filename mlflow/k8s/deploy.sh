#!/usr/bin/env bash
# deploy.sh — Full local MLflow + CNN Classifier deployment on Minikube (no cloud required)
# Usage: ./k8s/deploy.sh

set -euo pipefail

IMAGE_NAME="cnn-classifier"
IMAGE_TAG="latest"
NAMESPACE="cnn-classifier"
# MLflow runs as a ClusterIP service inside Minikube — no external server needed
MLFLOW_URI="http://mlflow-server:5000"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ─── 1. Prerequisites ──────────────────────────────────────────────────────────
echo "==> [1/8] Checking prerequisites..."
command -v minikube >/dev/null 2>&1 || { echo "ERROR: minikube not found. Install: https://minikube.sigs.k8s.io/docs/start/"; exit 1; }
command -v kubectl  >/dev/null 2>&1 || { echo "ERROR: kubectl not found.";  exit 1; }
command -v docker   >/dev/null 2>&1 || { echo "ERROR: docker not found.";   exit 1; }

# ─── 2. Start Minikube ─────────────────────────────────────────────────────────
echo "==> [2/8] Starting Minikube (if not already running)..."
MINIKUBE_HOST_STATUS=$(minikube status --format="{{.Host}}" 2>/dev/null || true)

if echo "$MINIKUBE_HOST_STATUS" | grep -q "Running"; then
  # Already running — verify the API server is actually reachable
  if ! kubectl cluster-info --request-timeout=10s >/dev/null 2>&1; then
    echo "    Minikube is running but the API server is unresponsive. Deleting and recreating..."
    minikube delete
    minikube start --memory=6144 --cpus=4 --driver=docker
  else
    echo "    Minikube is already running and healthy."
  fi
else
  # Not running — delete any stale profile to avoid memory-change errors, then start fresh
  minikube delete 2>/dev/null || true
  minikube start --memory=6144 --cpus=4 --driver=docker
fi

# Confirm API server is ready before continuing
kubectl cluster-info --request-timeout=30s >/dev/null

# ─── 3. Build app image inside Minikube ───────────────────────────────────────
echo "==> [3/8] Building Docker image inside Minikube daemon..."
# Changes are only visible to Minikube if the image is built inside its Docker daemon
eval "$(minikube docker-env)"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" "$PROJECT_DIR"
echo "    Image '${IMAGE_NAME}:${IMAGE_TAG}' built successfully."

# ─── 4. Apply namespace ────────────────────────────────────────────────────────
echo "==> [4/8] Applying namespace..."
kubectl apply -f "$SCRIPT_DIR/namespace.yaml"

# ─── 5. Deploy MLflow server ──────────────────────────────────────────────────
echo "==> [5/8] Deploying MLflow tracking server..."
kubectl apply -f "$SCRIPT_DIR/mlflow-pvc.yaml"
kubectl apply -f "$SCRIPT_DIR/mlflow-deployment.yaml"

echo "    Waiting for MLflow server to be ready..."
kubectl rollout status deployment/mlflow-server -n "$NAMESPACE" --timeout=180s

# ─── 6. Deploy app secret and PVC ─────────────────────────────────────────────
echo "==> [6/8] Applying Secret and PVC for the app..."
kubectl create secret generic cnn-classifier-secrets \
  --namespace="$NAMESPACE" \
  --from-literal=MLFLOW_TRACKING_URI="$MLFLOW_URI" \
  --save-config \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "$SCRIPT_DIR/pvc.yaml"

# ─── 7. Deploy CNN Classifier app ─────────────────────────────────────────────
echo "==> [7/8] Deploying CNN Classifier app..."
kubectl apply -f "$SCRIPT_DIR/deployment.yaml"
kubectl apply -f "$SCRIPT_DIR/service.yaml"

echo "    Waiting for app rollout (this may take a few minutes — TF loads the model)..."
kubectl rollout status deployment/cnn-classifier -n "$NAMESPACE" --timeout=300s

# ─── 8. Print URLs ────────────────────────────────────────────────────────────
echo ""
echo "==> [8/8] All deployments are live!"
echo ""
MINIKUBE_IP=$(minikube ip)
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  MLflow UI    : http://${MINIKUBE_IP}:30500              │"
echo "  │  App Home     : http://${MINIKUBE_IP}:30080              │"
echo "  │  Train route  : http://${MINIKUBE_IP}:30080/train  (POST)│"
echo "  │  Predict route: http://${MINIKUBE_IP}:30080/predict (POST)│"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""
echo "  Or open with tunnel:"
echo "    minikube service mlflow-server         -n ${NAMESPACE}"
echo "    minikube service cnn-classifier-service -n ${NAMESPACE}"
echo ""
echo "  To redeploy after code changes:"
echo "    eval \$(minikube docker-env)"
echo "    docker build -t cnn-classifier:latest ${PROJECT_DIR}"
echo "    kubectl rollout restart deployment/cnn-classifier -n ${NAMESPACE}"
echo ""
