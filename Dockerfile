# Inference service image. Build from the repo root:
#   docker build -t bioml-api .
#   docker run -p 8000:8000 -v $PWD/models/v2:/models/v2:ro bioml-api
# The model directory is mounted, not baked in, so the same image serves any
# model version whose metadata.json matches the installed feature contract.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY research/spec.yaml ./research/spec.yaml
RUN uv pip install --no-cache ".[ml,serve]"

ENV BIOML_MODEL_PATH=/models/v2
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "bioml.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
