FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy backend source and artifacts
COPY backend/ ./backend/
COPY notebookV3/checkpoints/ ./notebookV3/checkpoints/
COPY notebookV3/reports/ ./notebookV3/reports/

ENV PYTHONUNBUFFERED=1
ENV DATA_ROOT=/data
ENV MODEL_CHECKPOINT=/app/notebookV3/checkpoints/best_unet_kits23.pth

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
