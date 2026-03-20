#!/usr/bin/env bash
# =============================================================================
#  deploy.sh  –  Build the Docker image and deploy Agentic AI to Minikube
# =============================================================================
#
#  Prerequisites (must be installed and in PATH):
#    • minikube   https://minikube.sigs.k8s.io/docs/start/
#    • kubectl    https://kubernetes.io/docs/tasks/tools/
#    • docker     https://docs.docker.com/engine/install/
#
#  Required environment variable:
#    OPENAI_API_KEY   – your OpenAI secret key
#
#  Note on the RAG agent PDF:
#    The PDF (Stock_Market_Performance_2024.pdf) is bundled inside the Docker
#    image at /app/Agents/.  No external volume mount is required.
#
#  Usage:
#    export OPENAI_API_KEY="sk-..."
#    bash scripts/deploy.sh
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# 0. Validate prerequisites
# ---------------------------------------------------------------------------
info "Checking prerequisites..."
for cmd in minikube kubectl docker; do
    command -v "$cmd" &>/dev/null || error "'$cmd' is not installed or not in PATH."
done
success "All required tools found."

# ---------------------------------------------------------------------------
# 1. Verify the OpenAI API key is set
# ---------------------------------------------------------------------------
[[ -z "${OPENAI_API_KEY:-}" ]] && \
    error "OPENAI_API_KEY is not set.\n  Run: export OPENAI_API_KEY='sk-...'"
success "OPENAI_API_KEY is set."

# ---------------------------------------------------------------------------
# 2. Start Minikube (skip if already running)
# ---------------------------------------------------------------------------
info "Checking Minikube status..."
if ! minikube status --format='{{.Host}}' 2>/dev/null | grep -q "Running"; then
    info "Starting Minikube (driver=docker, memory=4096, cpus=2)..."
    minikube start --driver=docker --memory=4096 --cpus=2
else
    success "Minikube is already running."
fi

# ---------------------------------------------------------------------------
# 3. Point the local Docker CLI at Minikube's daemon
#    (images built here are immediately visible to Kubernetes)
# ---------------------------------------------------------------------------
info "Switching Docker context to Minikube daemon..."
eval "$(minikube docker-env)"
success "Docker context switched."

# ---------------------------------------------------------------------------
# 4. Build the Docker image inside Minikube
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

info "Building Docker image 'agentic-ai:latest' from $PROJECT_DIR ..."
docker build -t agentic-ai:latest "$PROJECT_DIR"
success "Docker image built successfully."

# ---------------------------------------------------------------------------
# 5. Apply Kubernetes manifests
# ---------------------------------------------------------------------------
K8S_DIR="$PROJECT_DIR/k8s"

info "Applying ConfigMap..."
kubectl apply -f "$K8S_DIR/configmap.yaml"

info "Creating / updating OpenAI API key Secret..."
kubectl create secret generic agentic-ai-secrets \
    --from-literal=openai-api-key="$OPENAI_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
success "Secret applied."

info "Applying Deployment..."
kubectl apply -f "$K8S_DIR/deployment.yaml"

info "Applying Service..."
kubectl apply -f "$K8S_DIR/service.yaml"

# ---------------------------------------------------------------------------
# 6. Wait for the deployment to become ready
# ---------------------------------------------------------------------------
info "Waiting for deployment rollout (up to 3 minutes)..."
kubectl rollout status deployment/agentic-ai --timeout=180s
success "Deployment is ready."

# ---------------------------------------------------------------------------
# 8. Print access information
# ---------------------------------------------------------------------------
SERVICE_URL=$(minikube service agentic-ai-service --url 2>/dev/null)

echo ""
echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  Agentic AI deployed successfully on Minikube!${NC}"
echo -e "${GREEN}================================================================${NC}"
echo ""
echo -e "  Base URL   : ${CYAN}${SERVICE_URL}${NC}"
echo -e "  Swagger UI : ${CYAN}${SERVICE_URL}/docs${NC}"
echo -e "  Health     : ${CYAN}${SERVICE_URL}/health${NC}"
echo ""
echo -e "  Quick tests:"
echo -e "    ${YELLOW}curl ${SERVICE_URL}/health${NC}"
echo -e "    ${YELLOW}curl -sX POST ${SERVICE_URL}/api/agent-bot/chat \\"
echo -e "         -H 'Content-Type: application/json' \\"
echo -e "         -d '{\"message\":\"Hello!\"}'${NC}"
echo ""
echo -e "  Watch logs:"
echo -e "    ${YELLOW}kubectl logs -f deployment/agentic-ai${NC}"
echo ""
echo -e "  Tear down:"
echo -e "    ${YELLOW}kubectl delete -f k8s/${NC}"
echo ""
