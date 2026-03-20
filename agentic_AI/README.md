# Agentic AI — LangGraph Agents on Minikube

A production-style deployment of five **LangGraph**-powered AI agents, served as a **FastAPI** REST API and deployed on **Minikube** (Kubernetes).  
All agents were originally interactive CLI scripts; they are now exposed as stateless/session-aware HTTP endpoints containerised with Docker and orchestrated with Kubernetes.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Agents — What They Do](#agents--what-they-do)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Deployment — Step-by-Step](#deployment--step-by-step)
- [Accessing the Running API](#accessing-the-running-api)
- [Verifying & Using Each Agent](#verifying--using-each-agent)
  - [1. Health Check](#1-health-check)
  - [2. Agent Bot](#2-agent-bot-stateless-chatbot)
  - [3. Memory Agent](#3-memory-agent-multi-turn-chat)
  - [4. ReAct Agent](#4-react-agent-math-tools)
  - [5. Drafter Agent](#5-drafter-agent-document-drafting)
  - [6. RAG Agent](#6-rag-agent-pdf-qa)
- [Swagger UI (Interactive Docs)](#swagger-ui-interactive-docs)
- [Operational Commands](#operational-commands)
- [Requirements](#requirements)

---

## Project Overview

**LangGraph** is a Python framework for building stateful, graph-structured AI agent workflows on top of LangChain. This project takes five LangGraph agent patterns and:

1. Wraps them in a **FastAPI** application (`app.py`) as REST endpoints.
2. Packages the application in a **Docker** multi-stage image.
3. Deploys it to a local **Minikube** Kubernetes cluster with a Deployment, NodePort Service, ConfigMap, and Secret.

The Stock Market Performance 2024 PDF (used by the RAG agent) is **bundled inside the Docker image** at `/app/Agents/`, so no external volume mount is required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Minikube Cluster                    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Deployment: agentic-ai                 │   │
│  │                                                  │   │
│  │  ┌────────────────────────────────────────────┐  │   │
│  │  │  Pod: agentic-ai (Python 3.11-slim)        │  │   │
│  │  │                                            │  │   │
│  │  │  FastAPI app (uvicorn :8000)               │  │   │
│  │  │  ├── /api/agent-bot/chat   → Agent Bot     │  │   │
│  │  │  ├── /api/memory/chat      → Memory Agent  │  │   │
│  │  │  ├── /api/react/solve      → ReAct Agent   │  │   │
│  │  │  ├── /api/drafter/chat     → Drafter Agent │  │   │
│  │  │  └── /api/rag/query        → RAG Agent     │  │   │
│  │  │                                            │  │   │
│  │  │  Env: OPENAI_API_KEY (from Secret)         │  │   │
│  │  │  Env: PDF_PATH, CHROMA_PERSIST_DIR         │  │   │
│  │  │       (from ConfigMap)                     │  │   │
│  │  └────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Service: agentic-ai-service (NodePort 30800)           │
└──────────────────────────┬──────────────────────────────┘
                           │ kubectl port-forward :8080
                    localhost:8080
                           │
                     Your Browser / curl
```

**Request flow:**  
`curl / browser` → `port-forward (8080)` → `Service (:80)` → `Pod (:8000)` → `FastAPI route` → `LangGraph agent` → `OpenAI API`

---

## Agents — What They Do

| Agent | Endpoint | Description | State |
|---|---|---|---|
| **Agent Bot** | `POST /api/agent-bot/chat` | Single-turn GPT-4o chatbot | Stateless |
| **Memory Agent** | `POST /api/memory/chat` | Multi-turn chat with full conversation history | Session-based |
| **ReAct Agent** | `POST /api/react/solve` | Solves math queries using add / subtract / multiply tools | Stateless |
| **Drafter Agent** | `POST /api/drafter/chat` | Iteratively writes and saves documents via LLM tool calls | Session-based |
| **RAG Agent** | `POST /api/rag/query` | Answers questions about the Stock Market 2024 PDF using ChromaDB retrieval | Stateless |

### How LangGraph Powers Each Agent

- **Agent Bot** — Linear graph: `START → process (LLM call) → END`
- **Memory Agent** — Direct LLM call with accumulated message list stored server-side per session
- **ReAct Agent** — Conditional graph: `agent ↔ tools` loop; terminates when no tool calls remain
- **Drafter Agent** — LLM with bound `update_document` / `save_document` tools; session tracks document content
- **RAG Agent** — Two-node conditional graph: `llm ↔ action (retriever_tool)`; ChromaDB built lazily from the bundled PDF

---

## Repository Structure

```
agentic_AI/
├── app.py                   # FastAPI app — all 5 agents as REST endpoints
├── Dockerfile               # Multi-stage Docker build (builder + slim runtime)
├── .dockerignore            # Excludes notebooks, caches, secrets from image
├── .env.example             # Template for local .env
├── requirements.txt         # Python dependencies
│
├── Agents/
│   ├── Agent_Bot.py         # Original CLI: stateless chatbot
│   ├── Memory_Agent.py      # Original CLI: chatbot with history
│   ├── ReAct.py             # Original CLI: math tool agent
│   ├── Drafter.py           # Original CLI: document drafting agent
│   ├── RAG_Agent.py         # Original CLI: PDF Q&A (fixed Windows path bug)
│   └── Stock_Market_Performance_2024.pdf   # Bundled PDF for RAG agent
│
├── k8s/
│   ├── deployment.yaml      # Kubernetes Deployment (1 replica, health probes)
│   ├── service.yaml         # NodePort Service (port 30800)
│   ├── configmap.yaml       # PDF_PATH and CHROMA_PERSIST_DIR config
│   └── secret.yaml          # Secret template (populated by deploy.sh)
│
├── scripts/
│   └── deploy.sh            # One-shot automated deployment script
│
├── Exercises/               # Jupyter notebooks — LangGraph exercise solutions
└── Graphs/                  # Jupyter notebooks — LangGraph concept demos
```

---

## Prerequisites

| Tool | Install guide |
|---|---|
| Docker | https://docs.docker.com/engine/install/ |
| Minikube | https://minikube.sigs.k8s.io/docs/start/ |
| kubectl | https://kubernetes.io/docs/tasks/tools/ |
| OpenAI API key | https://platform.openai.com/api-keys |

---

## Deployment — Step-by-Step

### Step 1 — Clone and enter the project

```bash
cd agentic_AI
```

### Step 2 — Add your OpenAI API key

```bash
echo 'OPENAI_API_KEY=sk-proj-your-real-key-here' > .env
```

### Step 3 — Run the deploy script (does everything automatically)

```bash
bash scripts/deploy.sh
```

The script:
1. Verifies prerequisites and the API key
2. Starts Minikube if not running (`--driver=docker --memory=4096 --cpus=2`)
3. Switches Docker context to Minikube's daemon
4. Builds `agentic-ai:latest` inside Minikube
5. Applies ConfigMap → Secret → Deployment → Service
6. Waits for the pod to reach `Running` state
7. Prints the service URL

### Step 4 — Expose the service (run once per terminal session)

```bash
kubectl port-forward service/agentic-ai-service 8080:80 &
```

The API is now accessible at **`http://localhost:8080`**.

---

## Accessing the Running API

| URL | Purpose |
|---|---|
| `http://localhost:8080/` | Root — lists all agents |
| `http://localhost:8080/health` | Liveness / readiness probe |
| `http://localhost:8080/docs` | Swagger UI — interactive API explorer |
| `http://localhost:8080/redoc` | ReDoc API documentation |

---

## Verifying & Using Each Agent

### 1. Health Check

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status": "healthy", "service": "Agentic AI API"}
```

---

### 2. Agent Bot (Stateless Chatbot)

Single-turn chat. No history is preserved between calls.

```bash
curl -s -X POST http://localhost:8080/api/agent-bot/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is LangGraph in one sentence?"}' | python3 -m json.tool
```

Expected response:
```json
{
    "response": "LangGraph is a Python framework ...",
    "session_id": "some-uuid"
}
```

---

### 3. Memory Agent (Multi-turn Chat)

Maintains full conversation history across calls within a session.

**First turn** (no `session_id` — creates a new session):
```bash
curl -s -X POST http://localhost:8080/api/memory/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My name is Amol"}' | python3 -m json.tool
```

Note the `session_id` in the response, then continue the conversation:

```bash
curl -s -X POST http://localhost:8080/api/memory/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my name?", "session_id": "<session_id_from_above>"}' \
  | python3 -m json.tool
```

View full conversation history:
```bash
curl http://localhost:8080/api/memory/sessions/<session_id> | python3 -m json.tool
```

Clear a session:
```bash
curl -X DELETE http://localhost:8080/api/memory/sessions/<session_id>
```

---

### 4. ReAct Agent (Math Tools)

Uses add / subtract / multiply tools in a ReAct reasoning loop.

```bash
curl -s -X POST http://localhost:8080/api/react/solve \
  -H "Content-Type: application/json" \
  -d '{"query": "Add 40 and 12, then multiply the result by 6"}' \
  | python3 -m json.tool
```

Expected response:
```json
{
    "result": "The result of adding 40 and 12 is 52, and multiplying that by 6 gives 312."
}
```

More examples:
```bash
# Chained operations
-d '{"query": "What is 100 minus 37, then subtract 8 more?"}'

# Mixed operations with explanation
-d '{"query": "Multiply 15 by 4, then add 20 to that result"}'
```

---

### 5. Drafter Agent (Document Drafting)

Creates and iteratively refines a document. The agent uses `update_document` and `save_document` LLM tool calls internally.

**Start a new document:**
```bash
curl -s -X POST http://localhost:8080/api/drafter/chat \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Write a professional introduction about AI in healthcare"}' \
  | python3 -m json.tool
```

Note the `session_id` and `document_content` in the response.

**Add more content to the same document:**
```bash
curl -s -X POST http://localhost:8080/api/drafter/chat \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Add a section on AI diagnostics with 2 bullet points", "session_id": "<session_id>"}' \
  | python3 -m json.tool
```

**Save the document:**
```bash
curl -s -X POST http://localhost:8080/api/drafter/chat \
  -H "Content-Type: application/json" \
  -d '{"instruction": "Save the document as healthcare_ai_report.txt", "session_id": "<session_id>"}' \
  | python3 -m json.tool
```

When saved, the response shows `"is_saved": true`.

---

### 6. RAG Agent (PDF Q&A)

Answers questions about the **Stock Market Performance 2024** PDF bundled in the image. ChromaDB is built on the first query (takes ~20s on first call; subsequent calls are fast).

```bash
curl -s -X POST http://localhost:8080/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How did the S&P 500 perform in 2024?"}' \
  | python3 -m json.tool
```

```bash
curl -s -X POST http://localhost:8080/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which sectors had the best performance in 2024?"}' \
  | python3 -m json.tool
```

Expected response format:
```json
{
    "answer": "According to the document (Chunk 1): The S&P 500 ..."
}
```

---

## Swagger UI (Interactive Docs)

Open **`http://localhost:8080/docs`** in your browser for a fully interactive API explorer where you can:
- See all endpoints with request/response schemas
- Send requests directly from the browser
- Inspect example payloads

---

## Operational Commands

```bash
# Check pod status
kubectl get pods -l app=agentic-ai

# Stream live logs
kubectl logs -f deployment/agentic-ai

# Describe pod (debug crash loops)
kubectl describe pod -l app=agentic-ai

# Re-expose after terminal restart
kubectl port-forward service/agentic-ai-service 8080:80 &

# Update API key and restart
export OPENAI_API_KEY="sk-..."
kubectl create secret generic agentic-ai-secrets \
    --from-literal=openai-api-key="$OPENAI_API_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/agentic-ai
kubectl rollout status deployment/agentic-ai

# Rebuild and redeploy after code changes
eval $(minikube docker-env)
docker build -t agentic-ai:latest .
kubectl rollout restart deployment/agentic-ai

# Scale replicas
kubectl scale deployment agentic-ai --replicas=2

# Tear down all Kubernetes resources
kubectl delete -f k8s/

# Stop Minikube
minikube stop
```

---

## Requirements

Core dependencies (`requirements.txt`):

```
langgraph
langchain
langchain-openai
langchain-community
langchain-chroma
chromadb
pypdf
fastapi
uvicorn[standard]
python-dotenv
```

Install locally (for development without Docker):
```bash
pip install -r requirements.txt
```

Run locally (without Docker):
```bash
python app.py
# API available at http://localhost:8000
```

