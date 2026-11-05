# AML WebSocket inference engine for Railway.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# System deps sometimes required by torch / scientific wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY aml ./aml
COPY artifacts/hetero_sage.pt artifacts/ibm_hi_small_demo.jsonl ./artifacts/

RUN uv sync --frozen --no-dev \
    && find /app/.venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

ENV PATH="/app/.venv/bin:$PATH"
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Railway injects PORT; aml-serve reads HOST/PORT from the environment.
EXPOSE 8080
CMD ["aml-serve", \
     "--host", "0.0.0.0", \
     "--checkpoint", "artifacts/hetero_sage.pt", \
     "--source", "sim", \
     "--replay", "artifacts/ibm_hi_small_demo.jsonl", \
     "--warmup", "200", \
     "--rate", "3"]
