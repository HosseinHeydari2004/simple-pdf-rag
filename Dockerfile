# ---------------------------------------------------------------------------
# DocuMind — Simple PDF RAG
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System deps needed by some document loaders (python-docx / unstructured
# style loaders occasionally shell out to these).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so this layer is cached across rebuilds
# that only touch application code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application.
COPY . .

# Where the Chroma vector store is persisted — mount this as a volume so the
# knowledge base survives container restarts.
RUN mkdir -p /app/vector_database

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.address=0.0.0.0", "--server.port=8501", \
            "--server.headless=true"]
