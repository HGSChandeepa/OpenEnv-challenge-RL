# server/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment code
COPY . /app/env

# Install Python dependencies
WORKDIR /app/env
RUN pip install --no-cache-dir -r server/requirements.txt
RUN pip install --no-cache-dir -e .

# Set PYTHONPATH
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server with extended WebSocket timeout for RL training
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "300", "--ws-ping-timeout", "300"]
