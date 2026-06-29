FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/cache/huggingface
ENV TORCH_HOME=/workspace/cache/torch
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# Docker builds don't have access to the GPU, so Torch can't auto-detect arch.
# 4090 = Ada (SM 8.9). Without this, CUDA extensions may compile without sm_89.
ENV TORCH_CUDA_ARCH_LIST=8.9

WORKDIR /workspace/Hunyuan3D-2

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

COPY . /workspace/Hunyuan3D-2

RUN python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN if [ -f requirements.txt ]; then python -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121; fi

# Install the package itself (gradio_app imports hy3dgen as a package)
RUN python -m pip install --no-cache-dir -e .

# Build required native extensions (CUDA + pybind11)
# These extension setup.py files import build-time deps (pybind11/torch) at import time.
# With PEP517, pip may build in an isolated env that doesn't have those deps installed yet.
RUN python -m pip install --no-cache-dir pybind11
RUN python -m pip install --no-cache-dir --no-build-isolation ./hy3dgen/texgen/differentiable_renderer
RUN python -m pip install --no-cache-dir --no-build-isolation ./hy3dgen/texgen/custom_rasterizer

EXPOSE 7860

ENTRYPOINT ["python", "gradio_app.py"]
CMD ["--host", "0.0.0.0", "--port", "7860", "--cache-path", "/workspace/outputs"]
