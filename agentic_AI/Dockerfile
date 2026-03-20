# ---- Build stage ------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies required for building Python packages with C extensions
# (chromadb, grpcio, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Runtime stage ----------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app.py .
COPY Agents/ ./Agents/

# Directories for optional PDF mount and ChromaDB persistence
RUN mkdir -p /app/data /app/chroma_db

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Use uvicorn directly – avoids spawning an extra shell process
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
