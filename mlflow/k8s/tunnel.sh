#!/usr/bin/env bash
# tunnel.sh — Forward Minikube services to localhost (required for WSL2)
# Run this after deploy.sh and keep it running in a terminal.
# Usage: ./k8s/tunnel.sh

set -euo pipefail

NAMESPACE="cnn-classifier"

echo ""
echo "  Starting port-forwards for WSL2 access..."
echo "  (Keep this terminal open — Ctrl+C to stop)"
echo ""

# Kill any previous port-forwards for these ports
pkill -f "port-forward.*mlflow-server" 2>/dev/null || true
pkill -f "port-forward.*cnn-classifier-service" 2>/dev/null || true
sleep 1

kubectl port-forward -n "$NAMESPACE" service/mlflow-server         5000:5000 --address=0.0.0.0 &
PF1=$!
kubectl port-forward -n "$NAMESPACE" service/cnn-classifier-service 8080:8080 --address=0.0.0.0 &
PF2=$!

sleep 3
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  MLflow UI    : http://localhost:5000            │"
echo "  │  App Home     : http://localhost:8080            │"
echo "  │  Train route  : http://localhost:8080/train      │"
echo "  │  Predict route: http://localhost:8080/predict    │"
echo "  └─────────────────────────────────────────────────┘"
echo ""

# Wait until one of the port-forwards exits (e.g. pod restart or Ctrl+C)
wait $PF1 $PF2
