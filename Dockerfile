FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /cache/huggingface \
    && chown -R app:app /cache /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/readyz || exit 1

CMD ["laya-server"]
