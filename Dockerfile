# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HY3DGEN_MODELS=/workspace/cache/hy3dgen
ENV HF_HOME=/workspace/cache/huggingface
ENV HF_HUB_CACHE=/workspace/cache/huggingface/hub
ENV HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface/hub
ENV TORCH_HOME=/workspace/cache/torch
ENV U2NET_HOME=/workspace/cache/u2net
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# Docker builds don't have access to the GPU, so Torch can't auto-detect arch.
# 4090 = Ada (SM 8.9). Without this, CUDA extensions may compile without sm_89.
ENV TORCH_CUDA_ARCH_LIST=8.9

WORKDIR /workspace/Hunyuan3D-2

# Pre-create cache/output directories (they can be mounted as volumes at runtime).
RUN mkdir -p \
    /workspace/cache/hy3dgen \
    /workspace/cache/huggingface/hub \
    /workspace/cache/torch \
    /workspace/cache/u2net \
    /workspace/outputs

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libgl1 \
    libglu1-mesa \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# Install Torch early so it stays cached unless Torch versions change.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy only dependency manifests first to maximize layer caching.
COPY requirements.txt /workspace/Hunyuan3D-2/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# Copy source (exclude large model cache from the app workdir; we copy it to /workspace/cache below).
COPY setup.py /workspace/Hunyuan3D-2/setup.py
COPY gradio_app.py /workspace/Hunyuan3D-2/gradio_app.py
COPY api_server.py /workspace/Hunyuan3D-2/api_server.py
COPY minimal_demo.py /workspace/Hunyuan3D-2/minimal_demo.py
COPY minimal_vae_demo.py /workspace/Hunyuan3D-2/minimal_vae_demo.py
COPY blender_addon.py /workspace/Hunyuan3D-2/blender_addon.py
COPY LICENSE /workspace/Hunyuan3D-2/LICENSE
COPY NOTICE /workspace/Hunyuan3D-2/NOTICE
COPY README.md /workspace/Hunyuan3D-2/README.md
COPY README_zh_cn.md /workspace/Hunyuan3D-2/README_zh_cn.md
COPY README_ja_jp.md /workspace/Hunyuan3D-2/README_ja_jp.md
COPY readme-docker.md /workspace/Hunyuan3D-2/readme-docker.md
COPY assets/ /workspace/Hunyuan3D-2/assets/
COPY docs/ /workspace/Hunyuan3D-2/docs/
COPY examples/ /workspace/Hunyuan3D-2/examples/
COPY hy3dgen/ /workspace/Hunyuan3D-2/hy3dgen/

# Install the package itself (gradio_app imports hy3dgen as a package)
RUN python -m pip install --no-cache-dir -e .

# Build required native extensions (CUDA + pybind11)
# These extension setup.py files import build-time deps (pybind11/torch) at import time.
# With PEP517, pip may build in an isolated env that doesn't have those deps installed yet.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pybind11
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-build-isolation ./hy3dgen/texgen/differentiable_renderer
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-build-isolation ./hy3dgen/texgen/custom_rasterizer

# Copy mirrored model cache from the repo into the runtime cache location.
# Populate this folder by running:
#   python gradio_app.py --prefetch_models --model-cache-dir cache
COPY cache/ /workspace/cache/

EXPOSE 7860

# Persist these across container restarts by mounting volumes.
VOLUME ["/workspace/cache", "/workspace/outputs"]

ENTRYPOINT ["python", "gradio_app.py"]
CMD ["--host", "0.0.0.0", "--port", "7860", "--cache-path", "/workspace/outputs", "--model-cache-dir", "/workspace/cache", "--lazy_load_models", "--idle_unload_sec", "600"]
