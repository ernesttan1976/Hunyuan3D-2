Run Hunyuan3D-2 in Docker on Windows (Docker Desktop + WSL2 + NVIDIA GPU).

## 1. Install prerequisites on Windows

Install:

* **NVIDIA Studio/Game Ready Driver**
* **Docker Desktop**
* **WSL2 backend enabled**
* **Git for Windows**

In Docker Desktop:

```text
Settings → General → Use the WSL 2 based engine
Settings → Resources → WSL Integration → Enable your Ubuntu distro
```

Test GPU access:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

You should see your **RTX 4090** listed.

---

## 2. Clone Hunyuan3D-2

From PowerShell:

```powershell
mkdir D:\AI
cd D:\AI
git clone https://github.com/Tencent/Hunyuan3D-2.git
cd Hunyuan3D-2
```

---

## 3. Create a Dockerfile

This repo already contains `Dockerfile` (and a `.dockerignore`).

---

## 4. Build the image

```powershell
docker build -t hunyuan3d-2 /d/Hunyuan3D-2
```

 DOCKER_BUILDKIT=1 docker build -t hunyuan3d-2 .  

---

## 5. Run Hunyuan3D-2 with GPU

```powershell
MSYS_NO_PATHCONV=1 docker run  --name hunyuan3d-2 --gpus all -d \
  -p 7860:7860 \
  --ipc=host \
  --env-file "D:/Hunyuan3D-2/.env" \
  -e HY3D_MAX_VRAM_GB=16 \
  -v /d/Hunyuan3D-2/outputs:/workspace/outputs \
  -v /d/Hunyuan3D-2/cache:/workspace/cache \
  hunyuan3d-2:test
```



Open:

```text
http://localhost:7860
```

---

## 6. Pick a model (optional)

The container defaults to `gradio_app.py` and exposes port `7860`.

To select a specific model/subfolder (examples from `README.md`):

```powershell
docker run --gpus all -it --rm -p 7860:7860 --ipc=host `
  -v D:\AI\Hunyuan3D-2\outputs:/workspace/outputs `
  -v D:\AI\Hunyuan3D-2\cache:/workspace/cache `
  hunyuan3d-2 `
  --model_path tencent/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0 --texgen_model_path tencent/Hunyuan3D-2 --low_vram_mode
```

---

## 7. Export to Blender / Unreal

Preferred path:

```text
Hunyuan3D-2 → GLB → Blender → cleanup/scale/origin → FBX → Unreal
```

In Blender:

```text
File → Import → glTF 2.0 (.glb)
File → Export → FBX
```

FBX export settings for Unreal:

```text
Forward: -Z Forward
Up: Y Up
Apply Transform: On
Path Mode: Copy
Embed Textures: On
```

---

## 8. Recommended folder layout

```text
D:\AI\
  Hunyuan3D-2\
    Dockerfile
    models\
    outputs\
    cache\
```

Your generated assets should appear in:

```text
D:\AI\Hunyuan3D-2\outputs
```

---

## 9. RTX 4090 notes

Use these settings when available:

```text
FP16 / half precision: enabled
Batch size: 1
Texture resolution: 1024 or 2048
Mesh simplification: enabled for Unreal
```

For Unreal, always inspect:

```text
polycount
UVs
normals
scale
material slots
collision
LODs
```

Best starting setup: **GLB from Hunyuan3D-2, Blender cleanup, FBX into Unreal**.



