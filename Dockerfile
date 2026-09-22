ARG CUDA_IMAGE=13.0.2-cudnn-runtime-ubuntu24.04
FROM nvidia/cuda:${CUDA_IMAGE}

ARG TORCH_VERSION=2.14.0
ARG TORCH_CUDA=cu130

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton \
    PATH=/opt/venv/bin:$PATH \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++

# Triton JIT compiles native launchers at inference time. Keep both the
# compiler toolchain and Python development headers in the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip \
        ca-certificates curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /cache/huggingface /cache/triton \
    && chown -R app:app /cache /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/readyz || exit 1

CMD ["laya-server"]
