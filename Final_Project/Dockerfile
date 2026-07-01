# IITK Agentic RAG — application image (API + ingestion + eval).
#
# torch is installed from the default PyPI wheel, which bundles CUDA — so the
# same image runs on CPU, and on GPU when the container is started with the
# NVIDIA runtime (`--gpus all` / compose `deploy.devices`). MODEL_DEVICE=auto
# then selects the GPU automatically.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System deps: curl for healthchecks/wait-for, libgomp1 for torch OpenMP,
# build-essential as a safety net for any sdist-only wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App source.
COPY . .

RUN chmod +x docker/entrypoint.sh

EXPOSE 8000

# entrypoint waits for backends, then runs whatever command the service passes
# (uvicorn for the API, run_ingestion for the ingest job, ragas_eval for eval).
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
