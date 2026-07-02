# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import random
import shutil
import time
import gc
import threading
import json
import struct
from glob import glob
from pathlib import Path

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uuid

from hy3dgen.shapegen.utils import logger

MAX_SEED = int(1e7)


# Lazily-initialized workers (models can consume significant CPU/GPU RAM).
# These globals are bound in __main__ and accessed via get_*() helpers.
_WORKER_LOCK = threading.RLock()
_LAST_USE_TS = 0.0
_ACTIVE_CALLS = 0
_LOADING_MODELS = 0

# Timing/state metadata for frontend status.
_LOADING_SINCE = None  # monotonic seconds
_LAST_LOADING_SEC = None
_STOPPING_MODELS = 0
_STOPPING_SINCE = None

i23d_worker = None
rmbg_worker = None
texgen_worker = None
t2i_worker = None

floater_remove_worker = None
degenerate_face_remove_worker = None
face_reduce_worker = None

# Lazy-imported symbols from hy3dgen
Hunyuan3DDiTFlowMatchingPipeline = None
FaceReducer = None
FloaterRemover = None
DegenerateFaceRemover = None
export_to_trimesh = None
BackgroundRemover = None
Hunyuan3DPaintPipeline = None
HunyuanDiTPipeline = None


def _touch_last_use() -> None:
    global _LAST_USE_TS
    _LAST_USE_TS = time.monotonic()


class _ActiveCall:
    def __enter__(self):
        global _ACTIVE_CALLS
        with _WORKER_LOCK:
            _ACTIVE_CALLS += 1
            _touch_last_use()

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_CALLS
        with _WORKER_LOCK:
            _ACTIVE_CALLS = max(0, _ACTIVE_CALLS - 1)
            _touch_last_use()


class _ModelLoading:
    """Marks model initialization as "active" so idle-unload doesn't race model load."""

    def __enter__(self):
        global _LOADING_MODELS
        global _LOADING_SINCE
        with _WORKER_LOCK:
            if _LOADING_MODELS == 0 and _LOADING_SINCE is None:
                _LOADING_SINCE = time.monotonic()
            _LOADING_MODELS += 1
            _touch_last_use()

    def __exit__(self, exc_type, exc, tb):
        global _LOADING_MODELS
        global _LOADING_SINCE, _LAST_LOADING_SEC
        with _WORKER_LOCK:
            _LOADING_MODELS = max(0, _LOADING_MODELS - 1)
            if _LOADING_MODELS == 0 and _LOADING_SINCE is not None:
                try:
                    _LAST_LOADING_SEC = max(0.0, time.monotonic() - _LOADING_SINCE)
                finally:
                    _LOADING_SINCE = None
            _touch_last_use()


class _ModelStopping:
    """Marks model shutdown/unload as "active" so status can reflect stopping."""

    def __enter__(self):
        global _STOPPING_MODELS, _STOPPING_SINCE
        with _WORKER_LOCK:
            if _STOPPING_MODELS == 0 and _STOPPING_SINCE is None:
                _STOPPING_SINCE = time.monotonic()
            _STOPPING_MODELS += 1

    def __exit__(self, exc_type, exc, tb):
        global _STOPPING_MODELS, _STOPPING_SINCE
        with _WORKER_LOCK:
            _STOPPING_MODELS = max(0, _STOPPING_MODELS - 1)
            if _STOPPING_MODELS == 0:
                _STOPPING_SINCE = None


def _lazy_import_shapegen() -> None:
    global Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover, export_to_trimesh
    if Hunyuan3DDiTFlowMatchingPipeline is not None:
        return
    from hy3dgen.shapegen import (
        FaceReducer as _FaceReducer,
        FloaterRemover as _FloaterRemover,
        DegenerateFaceRemover as _DegenerateFaceRemover,
        Hunyuan3DDiTFlowMatchingPipeline as _Hunyuan3DDiTFlowMatchingPipeline,
    )
    from hy3dgen.shapegen.pipelines import export_to_trimesh as _export_to_trimesh

    FaceReducer = _FaceReducer
    FloaterRemover = _FloaterRemover
    DegenerateFaceRemover = _DegenerateFaceRemover
    Hunyuan3DDiTFlowMatchingPipeline = _Hunyuan3DDiTFlowMatchingPipeline
    export_to_trimesh = _export_to_trimesh


def _lazy_import_rembg() -> None:
    global BackgroundRemover
    if BackgroundRemover is not None:
        return
    from hy3dgen.rembg import BackgroundRemover as _BackgroundRemover

    BackgroundRemover = _BackgroundRemover


def _lazy_import_texgen() -> None:
    global Hunyuan3DPaintPipeline
    if Hunyuan3DPaintPipeline is not None:
        return
    from hy3dgen.texgen import Hunyuan3DPaintPipeline as _Hunyuan3DPaintPipeline

    Hunyuan3DPaintPipeline = _Hunyuan3DPaintPipeline


def _lazy_import_t2i() -> None:
    global HunyuanDiTPipeline
    if HunyuanDiTPipeline is not None:
        return
    from hy3dgen.text2image import HunyuanDiTPipeline as _HunyuanDiTPipeline

    HunyuanDiTPipeline = _HunyuanDiTPipeline


def unload_models(reason: str = "") -> None:
    """Best-effort release of model RAM/VRAM. Safe to call repeatedly."""
    global i23d_worker, rmbg_worker, texgen_worker, t2i_worker
    global floater_remove_worker, degenerate_face_remove_worker, face_reduce_worker

    with _ModelStopping():
        with _WORKER_LOCK:
            if _ACTIVE_CALLS != 0:
                return

            i23d_worker = None
            rmbg_worker = None
            texgen_worker = None
            t2i_worker = None

            floater_remove_worker = None
            degenerate_face_remove_worker = None
            face_reduce_worker = None

        # Make sure references are gone before collecting.
        gc.collect()

        # For CUDA, this drops PyTorch's caching allocator blocks back to the driver.
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        if reason:
            print(f"[idle-unload] Unloaded models ({reason}).")


def _get_model_status_snapshot() -> dict:
    """Thread-safe snapshot for UI polling and external status endpoint."""
    now = time.monotonic()
    with _WORKER_LOCK:
        any_loaded = any(
            x is not None
            for x in (
                i23d_worker,
                rmbg_worker,
                texgen_worker,
                t2i_worker,
                floater_remove_worker,
                degenerate_face_remove_worker,
                face_reduce_worker,
            )
        )

        if _STOPPING_MODELS > 0:
            status = "stopping"
        elif _LOADING_MODELS > 0:
            status = "loading"
        elif any_loaded:
            status = "loaded"
        else:
            status = "not loaded"

        idle_for_sec = (now - _LAST_USE_TS) if _LAST_USE_TS else 0.0

        # Always compute a countdown; UI decides how to render when disabled.
        shutdown_in_sec = None
        try:
            idle_unload_sec = float(getattr(args, "idle_unload_sec", 0.0) or 0.0)
        except Exception:
            idle_unload_sec = 0.0
        if idle_unload_sec > 0:
            shutdown_in_sec = max(0.0, idle_unload_sec - idle_for_sec)

        loading_for_sec = (now - _LOADING_SINCE) if _LOADING_SINCE else None
        last_loading_sec = _LAST_LOADING_SEC

        return {
            "status": status,
            "idle_for_sec": idle_for_sec,
            "shutdown_in_sec": shutdown_in_sec,
            "loading_for_sec": loading_for_sec,
            "last_loading_sec": last_loading_sec,
        }


def _format_model_status_html(s: dict) -> str:
    status = (s or {}).get("status") or "not loaded"

    shutdown_in = (s or {}).get("shutdown_in_sec")
    if shutdown_in is None:
        shutdown_msg = "Warning: model shutdown in disabled"
    else:
        shutdown_msg = f"Warning: model shutdown in {int(round(max(0.0, shutdown_in)))} seconds"

    loading_for = (s or {}).get("loading_for_sec")
    last_loading = (s or {}).get("last_loading_sec")

    loading_msg = None
    if status == "loading" and loading_for is not None:
        loading_msg = f"Warning: model loading takes {int(round(max(0.0, loading_for)))} seconds"
        if last_loading is not None:
            loading_msg += f" (estimated ~{int(round(max(0.0, last_loading)))} seconds)"
    else:
        if last_loading is not None:
            loading_msg = f"Warning: model loading takes ~{int(round(max(0.0, last_loading)))} seconds"

    badge_color = {
        "not loaded": "#6b7280",
        "loading": "#b45309",
        "loaded": "#065f46",
        "stopping": "#b91c1c",
    }.get(status, "#6b7280")

    parts = [
        f"<span style='font-weight:600'>Model status:</span> "
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{badge_color};color:white;font-size:12px;line-height:18px'>"
        f"{status}</span>",
        f"<span style='color:#b45309;font-weight:600'>{shutdown_msg}</span>",
    ]
    if loading_msg:
        parts.insert(1, f"<span style='color:#b45309;font-weight:600'>{loading_msg}</span>")

    return (
        "<div style='margin: 6px 0 10px 0; display:flex; gap:12px; flex-wrap:wrap; align-items:center'>"
        + "".join(parts)
        + "</div>"
    )


def get_export_to_trimesh():
    _lazy_import_shapegen()
    return export_to_trimesh


def get_rmbg_worker():
    global rmbg_worker
    with _WORKER_LOCK:
        _touch_last_use()
        if rmbg_worker is None:
            _lazy_import_rembg()
            with _ModelLoading():
                rmbg_worker = BackgroundRemover()
        return rmbg_worker


def get_i23d_worker():
    global i23d_worker
    with _WORKER_LOCK:
        _touch_last_use()
        if i23d_worker is None:
            _lazy_import_shapegen()
            with _ModelLoading():
                i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    args.model_path,
                    subfolder=args.subfolder,
                    use_safetensors=True,
                    device=args.device,
                )
                if args.enable_flashvdm:
                    mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
                    i23d_worker.enable_flashvdm(mc_algo=mc_algo)
                if args.compile:
                    i23d_worker.compile()
        return i23d_worker


def get_floater_remove_worker():
    global floater_remove_worker
    with _WORKER_LOCK:
        _touch_last_use()
        if floater_remove_worker is None:
            _lazy_import_shapegen()
            with _ModelLoading():
                floater_remove_worker = FloaterRemover()
        return floater_remove_worker


def get_degenerate_face_remove_worker():
    global degenerate_face_remove_worker
    with _WORKER_LOCK:
        _touch_last_use()
        if degenerate_face_remove_worker is None:
            _lazy_import_shapegen()
            with _ModelLoading():
                degenerate_face_remove_worker = DegenerateFaceRemover()
        return degenerate_face_remove_worker


def get_face_reduce_worker():
    global face_reduce_worker
    with _WORKER_LOCK:
        _touch_last_use()
        if face_reduce_worker is None:
            _lazy_import_shapegen()
            with _ModelLoading():
                face_reduce_worker = FaceReducer()
        return face_reduce_worker


def get_texgen_worker():
    global texgen_worker
    if args.disable_tex:
        raise gr.Error("Texture synthesis is disabled (started with --disable_tex).")
    with _WORKER_LOCK:
        _touch_last_use()
        if texgen_worker is None:
            _lazy_import_texgen()
            with _ModelLoading():
                texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
                if args.low_vram_mode:
                    texgen_worker.enable_model_cpu_offload()
        return texgen_worker


def get_t2i_worker():
    global t2i_worker
    if not args.enable_t23d:
        raise gr.Error("Text to 3D is disable. To activate it, please run `python gradio_app.py --enable_t23d`.")
    with _WORKER_LOCK:
        _touch_last_use()
        if t2i_worker is None:
            _lazy_import_t2i()
            with _ModelLoading():
                t2i_worker = HunyuanDiTPipeline(
                    'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                    device=args.device,
                )
        return t2i_worker


def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def get_example_mv_list():
    print('Loading example mv list ...')
    mv_list = list()
    root = './assets/example_mv_images'
    for mv_dir in os.listdir(root):
        view_list = []
        for view in ['front', 'back', 'left', 'right']:
            path = os.path.join(root, mv_dir, f'{view}.png')
            if os.path.exists(path):
                view_list.append(path)
            else:
                view_list.append(None)
        mv_list.append(view_list)
    return mv_list


def gen_save_folder(max_size=200):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 获取所有文件夹路径
    # Never delete reserved static assets (served by model-viewer templates).
    reserved = {"env_maps"}
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir() and f.name not in reserved]

    # 如果文件夹数量超过 max_size，删除创建时间最久的文件夹
    if len(dirs) >= max_size:
        # 按创建时间排序，最久的排在前面
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")

    # 生成一个新的 uuid 文件夹名称
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")

    return new_folder


def export_mesh(mesh, save_folder, textured=False, type='glb'):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        # model-viewer expects a file URL; do not append a trailing slash.
        template_html = template_html.replace('#src#', f'{related_path}')
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    # Use URL separators even on Windows.
    rel_path = rel_path.replace('\\', '/')
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(
        f'Find html file {output_html_path}, {os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """


def _gen_shape(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    if not MV_MODE and image is None and caption is None:
        raise gr.Error("Please provide either a caption or an image.")

    if MV_MODE:
        if mv_image_front is None and mv_image_back is None and mv_image_left is None and mv_image_right is None:
            raise gr.Error("Please provide at least one view image.")
        image = {}
        if mv_image_front:
            image['front'] = mv_image_front
        if mv_image_back:
            image['back'] = mv_image_back
        if mv_image_left:
            image['left'] = mv_image_left
        if mv_image_right:
            image['right'] = mv_image_right

    seed = int(randomize_seed_fn(seed, randomize_seed))

    octree_resolution = int(octree_resolution)
    if caption:
        print('prompt is', caption)

    save_folder = gen_save_folder()
    stats = {
        'model': {
            'shapegen': f'{args.model_path}/{args.subfolder}',
            'texgen': f'{args.texgen_model_path}',
        },
        'params': {
            'caption': caption,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
        }
    }
    time_meta = {}

    def _save_input_png(pil_img) -> None:
        # Best-effort: sessions should still succeed even if disk write fails.
        try:
            if pil_img is None:
                return
            out_path = os.path.join(save_folder, 'input.png')
            # Preserve transparency if present.
            pil_img.save(out_path)
        except Exception as e:
            print(f"Failed to save input.png: {e}")

    if image is None:
        start_time = time.time()
        image = get_t2i_worker()(caption)
        time_meta['text2image'] = time.time() - start_time

    # Persist the original input image for session browsing/replay.
    if not MV_MODE:
        _save_input_png(image)

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'input.png'))

    _rmbg = get_rmbg_worker()
    if MV_MODE:
        start_time = time.time()
        for k, v in image.items():
            if check_box_rembg or v.mode == "RGB":
                img = _rmbg(v.convert('RGB'))
                image[k] = img
        time_meta['remove background'] = time.time() - start_time
    else:
        if check_box_rembg or image.mode == "RGB":
            start_time = time.time()
            image = _rmbg(image.convert('RGB'))
            time_meta['remove background'] = time.time() - start_time

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'rembg.png'))

    # image to white model
    start_time = time.time()
    generator = torch.Generator().manual_seed(int(seed))
    outputs = get_i23d_worker()(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh'
    )
    time_meta['shape generation'] = time.time() - start_time
    logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))

    tmp_start = time.time()
    mesh = get_export_to_trimesh()(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    stats['time'] = time_meta
    main_image = image if not MV_MODE else image['front']
    return mesh, main_image, save_folder, stats, seed


def generation_all(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    with _ActiveCall():
        start_time_0 = time.time()
        mesh, image, save_folder, stats, seed = _gen_shape(
            caption,
            image,
            mv_image_front=mv_image_front,
            mv_image_back=mv_image_back,
            mv_image_left=mv_image_left,
            mv_image_right=mv_image_right,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            octree_resolution=octree_resolution,
            check_box_rembg=check_box_rembg,
            num_chunks=num_chunks,
            randomize_seed=randomize_seed,
        )
        path = export_mesh(mesh, save_folder, textured=False)

    # tmp_time = time.time()
    # mesh = floater_remove_worker(mesh)
    # mesh = degenerate_face_remove_worker(mesh)
    # logger.info("---Postprocessing takes %s seconds ---" % (time.time() - tmp_time))
    # stats['time']['postprocessing'] = time.time() - tmp_time

        tmp_time = time.time()
        mesh = get_face_reduce_worker()(mesh)
        logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
        stats['time']['face reduction'] = time.time() - tmp_time

        tmp_time = time.time()
        textured_mesh = get_texgen_worker()(mesh, image)
        logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
        stats['time']['texture generation'] = time.time() - tmp_time
        stats['time']['total'] = time.time() - start_time_0

        textured_mesh.metadata['extras'] = stats
        path_textured = export_mesh(textured_mesh, save_folder, textured=True)
        model_viewer_html_textured = build_model_viewer_html(
            save_folder,
            height=HTML_HEIGHT,
            width=HTML_WIDTH,
            textured=True,
        )
        if args.low_vram_mode:
            torch.cuda.empty_cache()
        return (
            gr.update(value=path),
            gr.update(value=path_textured),
            model_viewer_html_textured,
            stats,
            seed,
        )


def shape_generation(
    caption=None,
    image=None,
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    with _ActiveCall():
        start_time_0 = time.time()
        mesh, image, save_folder, stats, seed = _gen_shape(
            caption,
            image,
            mv_image_front=mv_image_front,
            mv_image_back=mv_image_back,
            mv_image_left=mv_image_left,
            mv_image_right=mv_image_right,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            octree_resolution=octree_resolution,
            check_box_rembg=check_box_rembg,
            num_chunks=num_chunks,
            randomize_seed=randomize_seed,
        )
        stats['time']['total'] = time.time() - start_time_0
        mesh.metadata['extras'] = stats

        path = export_mesh(mesh, save_folder, textured=False)
        model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
        if args.low_vram_mode:
            torch.cuda.empty_cache()
        return (
            gr.update(value=path),
            model_viewer_html,
            stats,
            seed,
        )


def build_app():
    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if MV_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'
    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">

    {title}
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    <div align="center">
      <a href="https://github.com/tencent/Hunyuan3D-2">Github</a> &ensp; 
      <a href="http://3d-models.hunyuan.tencent.com">Homepage</a> &ensp;
      <a href="https://3d.hunyuan.tencent.com">Hunyuan3D Studio</a> &ensp;
      <a href="#">Technical Report</a> &ensp;
      <a href="https://huggingface.co/Tencent/Hunyuan3D-2"> Pretrained Models</a> &ensp;
    </div>
    """
    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1480px;
    }
    .mv-image button .wrap {
        font-size: 10px;
    }

    .mv-image .icon-wrap {
        width: 20px;
    }

    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0', analytics_enabled=False, css=custom_css) as demo:
        gr.HTML(title_html)

        model_status_html = gr.HTML(_format_model_status_html(_get_model_status_snapshot()))

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt', visible=not MV_MODE) as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)

                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='HunyuanDiT will be used to generate image.',
                                             info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('MultiView Prompt', visible=MV_MODE) as tab_mv:
                        # gr.Label('Please upload at least one front image.')
                        with gr.Row():
                            mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')

                with gr.Row():
                    btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                    btn_all = gr.Button(value='Gen Textured Shape',
                                        variant='primary',
                                        visible=HAS_TEXTUREGEN,
                                        min_width=100)
                    btn_sessions = gr.Button(value='Sessions', min_width=100)

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                    with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(label='Generation Mode',
                                            info='Recommendation: Turbo for most cases, Fast for very complex cases, Standard seldom use.',
                                            choices=['Turbo', 'Fast', 'Standard'], value='Turbo')
                        decode_mode = gr.Radio(label='Decoding Mode',
                                               info='The resolution for exporting mesh from generated vectset',
                                               choices=['Low', 'Standard', 'High'],
                                               value='Standard')
                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background', min_width=100)
                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True, min_width=100)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                        )
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps')
                            octree_resolution = gr.Slider(maximum=512, minimum=16, value=256, label='Octree Resolution')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100)
                    with gr.Tab("Export", id='tab_export'):
                        with gr.Row():
                            file_type = gr.Dropdown(label='File Type', choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100)
                            reduce_face = gr.Checkbox(label='Simplify Mesh', value=False, min_width=100)
                            export_texture = gr.Checkbox(label='Include Texture', value=False,
                                                         visible=False, min_width=100)
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000,
                                                    label='Target Face Number')
                        with gr.Row():
                            confirm_export = gr.Button(value="Transform", min_width=100)
                            file_export = gr.DownloadButton(label="Download", variant='primary',
                                                            interactive=False, min_width=100)

            with gr.Column(scale=6):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')

            with gr.Column(scale=3 if MV_MODE else 2):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    # Put Sessions first so it doesn't get hidden behind tab overflow on narrow screens.
                    with gr.Tab('Sessions', id='tab_sessions'):
                        with gr.Row():
                            refresh_sessions = gr.Button(value='Refresh', min_width=100)
                        sessions_dropdown = gr.Dropdown(label='Past Sessions', choices=[], value=None)
                        session_input_preview = gr.Image(label='Saved Input (input.png)', type='filepath', height=290)

                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery', visible=not MV_MODE) as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=18)

                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery', visible=HAS_T2I and not MV_MODE) as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label=None, examples_per_page=18)
                    with gr.Tab('MultiView to 3D Gallery', id='tab_mv_gallery', visible=MV_MODE) as tab_mv:
                        with gr.Row():
                            gr.Examples(examples=example_mvs,
                                        inputs=[mv_image_front, mv_image_back, mv_image_left, mv_image_right],
                                        label=None, examples_per_page=6)

        btn_sessions.click(fn=lambda: gr.update(selected='tab_sessions'), outputs=[gallery])

        gr.HTML(f"""
        <div align="center">
        Activated Model - Shape Generation ({args.model_path}/{args.subfolder}) ; Texture Generation ({'Hunyuan3D-2' if HAS_TEXTUREGEN else 'Unavailable'})
        </div>
        """)
        if not HAS_TEXTUREGEN:
            gr.HTML("""
            <div style="margin-top: 5px;"  align="center">
                <b>Warning: </b>
                Texture synthesis is disable due to missing requirements,
                 please install requirements following <a href="https://github.com/Tencent/Hunyuan3D-2?tab=readme-ov-file#install-requirements">README.md</a>to activate it.
            </div>
            """)
        if not args.enable_t23d:
            gr.HTML("""
            <div style="margin-top: 5px;"  align="center">
                <b>Warning: </b>
                Text to 3D is disable. To activate it, please run `python gradio_app.py --enable_t23d`.
            </div>
            """)

        tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
        if HAS_T2I:
            tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed]
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value == 'Fast':
                return gr.update(value=10)
            else:
                return gr.update(value=30)

        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value == 'Low':
                return gr.update(value=196)
            elif value == 'Standard':
                return gr.update(value=256)
            else:
                return gr.update(value=384)

        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], outputs=[octree_resolution])

        def on_export_click(file_out, file_out2, file_type, reduce_face, export_texture, target_face_num):
            with _ActiveCall():
                if export_texture:
                    if file_out2 is None:
                        raise gr.Error('Please generate/load a textured mesh first.')
                else:
                    if file_out is None:
                        raise gr.Error('Please generate/load a mesh first.')

                print(f'exporting {file_out}')
                print(f'reduce face to {target_face_num}')
                if export_texture:
                    mesh = trimesh.load(file_out2)
                    save_folder = gen_save_folder()
                    path = export_mesh(mesh, save_folder, textured=True, type=file_type)

                    # for preview
                    save_folder = gen_save_folder()
                    _ = export_mesh(mesh, save_folder, textured=True)
                    model_viewer_html = build_model_viewer_html(
                        save_folder,
                        height=HTML_HEIGHT,
                        width=HTML_WIDTH,
                        textured=True,
                    )
                else:
                    mesh = trimesh.load(file_out)
                    mesh = get_floater_remove_worker()(mesh)
                    mesh = get_degenerate_face_remove_worker()(mesh)
                    if reduce_face:
                        mesh = get_face_reduce_worker()(mesh, target_face_num)
                    save_folder = gen_save_folder()
                    path = export_mesh(mesh, save_folder, textured=False, type=file_type)

                    # for preview
                    save_folder = gen_save_folder()
                    _ = export_mesh(mesh, save_folder, textured=False)
                    model_viewer_html = build_model_viewer_html(
                        save_folder,
                        height=HTML_HEIGHT,
                        width=HTML_WIDTH,
                        textured=False,
                    )
                print(f'export to {path}')
                return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected='export_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )

        # -----------------
        # Sessions Browser
        # -----------------

        def _read_textured_stats_from_glb(glb_path: str) -> dict:
            """Read embedded stats from textured_mesh.glb without importing trimesh."""
            try:
                b = Path(glb_path).read_bytes()
                # GLB header: magic(4) version(4) length(4)
                if len(b) < 20:
                    return {}
                magic, _version, _length = struct.unpack_from('<4sII', b, 0)
                if magic != b'glTF':
                    return {}
                chunk_len, chunk_type = struct.unpack_from('<I4s', b, 12)
                if chunk_type != b'JSON':
                    return {}
                j = json.loads(b[20:20 + chunk_len].decode('utf-8'))
                # Known layout from current exporter: meshes[0].extras.extras
                ex = (j.get('meshes') or [{}])[0].get('extras')
                if isinstance(ex, dict) and isinstance(ex.get('extras'), dict):
                    ex = ex.get('extras')
                if isinstance(ex, dict) and 'params' in ex and 'time' in ex:
                    return ex
            except Exception:
                return {}
            return {}

        def _list_sessions() -> tuple[list[tuple[str, str]], str | None]:
            root = Path(SAVE_DIR)
            if not root.exists():
                return [], None

            items: list[tuple[float, str, str]] = []
            for d in root.iterdir():
                if not d.is_dir() or d.name == 'env_maps':
                    continue
                textured = d / 'textured_mesh.glb'
                white = d / 'white_mesh.glb'
                if not textured.is_file() and not white.is_file():
                    continue

                mtime = d.stat().st_mtime
                seed = None
                steps = None
                if textured.is_file():
                    st = _read_textured_stats_from_glb(str(textured))
                    params = st.get('params') if isinstance(st, dict) else None
                    if isinstance(params, dict):
                        seed = params.get('seed')
                        steps = params.get('steps')

                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                short = d.name.split('-')[0]
                label = f"{ts} | {short}"
                if seed is not None:
                    label += f" | seed={seed}"
                if steps is not None:
                    label += f" | steps={steps}"
                items.append((mtime, label, d.name))

            items.sort(key=lambda t: t[0], reverse=True)
            choices = [(label, session_id) for _mtime, label, session_id in items]
            default_val = choices[0][1] if choices else None
            return choices, default_val

        def _load_session(session_id: str):
            if not session_id:
                return (
                    gr.update(value=None),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    HTML_OUTPUT_PLACEHOLDER,
                    HTML_OUTPUT_PLACEHOLDER,
                    {},
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(selected='gen_mesh_panel'),
                )

            folder = Path(SAVE_DIR) / session_id
            input_png = folder / 'input.png'
            textured_glb = folder / 'textured_mesh.glb'
            white_glb = folder / 'white_mesh.glb'

            # Prefer textured for preview/export when present.
            has_textured = textured_glb.is_file()
            has_white = white_glb.is_file()

            if has_textured:
                html_path = folder / 'textured_mesh.html'
                if not html_path.is_file():
                    _ = build_model_viewer_html(str(folder), height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
                viewer_html = build_model_viewer_html(str(folder), height=HTML_HEIGHT, width=HTML_WIDTH, textured=True)
                file_out_val = str(white_glb) if has_white else str(textured_glb)
                file_out2_val = str(textured_glb)
                st = _read_textured_stats_from_glb(str(textured_glb))
            elif has_white:
                html_path = folder / 'white_mesh.html'
                if not html_path.is_file():
                    _ = build_model_viewer_html(str(folder), height=HTML_HEIGHT, width=HTML_WIDTH, textured=False)
                viewer_html = build_model_viewer_html(str(folder), height=HTML_HEIGHT, width=HTML_WIDTH, textured=False)
                file_out_val = str(white_glb)
                file_out2_val = None
                st = {}
            else:
                viewer_html = HTML_OUTPUT_PLACEHOLDER
                file_out_val = None
                file_out2_val = None
                st = {}

            params = st.get('params') if isinstance(st, dict) else None
            if not isinstance(params, dict):
                params = {}

            seed_val = params.get('seed')
            caption_val = params.get('caption')

            # Restore controls for reproducibility; keep randomize_seed off.
            steps_val = params.get('steps')
            cfg_val = params.get('guidance_scale')
            oct_val = params.get('octree_resolution')
            rembg_val = params.get('check_box_rembg')
            chunks_val = params.get('num_chunks')

            # Input image: only update if present (older sessions may not have it).
            if input_png.is_file():
                img_update = gr.update(value=str(input_png))
                preview_update = gr.update(value=str(input_png))
                tab_prompt_update = gr.update(selected='tab_img_prompt')
            else:
                img_update = gr.update()
                preview_update = gr.update(value=None)
                tab_prompt_update = gr.update()

            # Export UX should match freshly-generated state.
            if has_textured:
                export_texture_update = gr.update(visible=True, value=True)
                reduce_face_update = gr.update(interactive=False)
            else:
                export_texture_update = gr.update(visible=False, value=False)
                reduce_face_update = gr.update(interactive=True)

            return (
                preview_update,
                img_update,
                gr.update(value=file_out_val),
                gr.update(value=file_out2_val),
                viewer_html,
                HTML_OUTPUT_PLACEHOLDER,
                st,
                gr.update(value=seed_val) if seed_val is not None else gr.update(),
                gr.update(value=caption_val) if caption_val is not None else gr.update(),
                gr.update(value=steps_val) if steps_val is not None else gr.update(),
                gr.update(value=cfg_val) if cfg_val is not None else gr.update(),
                gr.update(value=oct_val) if oct_val is not None else gr.update(),
                gr.update(value=rembg_val) if rembg_val is not None else gr.update(),
                gr.update(value=chunks_val) if chunks_val is not None else gr.update(),
                gr.update(value=False),
                export_texture_update,
                reduce_face_update,
                gr.update(interactive=True),
                gr.update(interactive=False),
                tab_prompt_update,
                gr.update(selected='gen_mesh_panel'),
            )

        def _refresh_sessions():
            choices, default_val = _list_sessions()
            return gr.update(choices=choices, value=default_val)

        refresh_sessions.click(_refresh_sessions, outputs=[sessions_dropdown]).then(
            _load_session,
            inputs=[sessions_dropdown],
            outputs=[
                session_input_preview,
                image,
                file_out,
                file_out2,
                html_gen_mesh,
                html_export_mesh,
                stats,
                seed,
                caption,
                num_steps,
                cfg_scale,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                export_texture,
                reduce_face,
                confirm_export,
                file_export,
                tabs_prompt,
                tabs_output,
            ],
        )

        sessions_dropdown.change(
            _load_session,
            inputs=[sessions_dropdown],
            outputs=[
                session_input_preview,
                image,
                file_out,
                file_out2,
                html_gen_mesh,
                html_export_mesh,
                stats,
                seed,
                caption,
                num_steps,
                cfg_scale,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                export_texture,
                reduce_face,
                confirm_export,
                file_export,
                tabs_prompt,
                tabs_output,
            ],
        )

        # Populate session list once on load (and load most recent).
        demo.load(_refresh_sessions, outputs=[sessions_dropdown]).then(
            _load_session,
            inputs=[sessions_dropdown],
            outputs=[
                session_input_preview,
                image,
                file_out,
                file_out2,
                html_gen_mesh,
                html_export_mesh,
                stats,
                seed,
                caption,
                num_steps,
                cfg_scale,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
                export_texture,
                reduce_face,
                confirm_export,
                file_export,
                tabs_prompt,
                tabs_output,
            ],
        )

        # Poll backend status every 5 seconds.
        def _poll_model_status():
            return _format_model_status_html(_get_model_status_snapshot())

        # Prefer a native periodic load; fall back to Timer if needed.
        try:
            demo.load(_poll_model_status, outputs=[model_status_html], every=5)
        except TypeError:
            try:
                t = gr.Timer(5)
                t.tick(_poll_model_status, outputs=[model_status_html])
            except Exception:
                # Last resort: at least populate once.
                demo.load(_poll_model_status, outputs=[model_status_html])

    return demo


if __name__ == '__main__':
    import argparse

    def _load_dotenv_if_present(dotenv_path: Path) -> None:
        """Minimal .env loader (no external deps).

        Only fills missing environment variables (does not override existing ones).
        """
        try:
            if not dotenv_path.is_file():
                return
            for raw_line in dotenv_path.read_text(encoding='utf-8').splitlines():
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[len('export '):].lstrip()
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if (
                    len(value) >= 2
                    and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'"))
                ):
                    value = value[1:-1]
                os.environ.setdefault(key, value)
        except Exception as e:
            print(f"Failed to read .env from {dotenv_path}: {e}")

    def _env_float(name: str) -> float | None:
        val = os.getenv(name)
        if val is None or val == '':
            return None
        try:
            return float(val)
        except ValueError:
            print(f"Invalid {name}={val!r}; ignoring.")
            return None

    # Load .env from the same folder as this script (repo root in this project).
    _load_dotenv_if_present(Path(__file__).with_name('.env'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='outputs')
    parser.add_argument(
        '--model-cache-dir',
        '--model_cache_dir',
        dest='model_cache_dir',
        type=str,
        default='cache',
        help='Persistent directory for model weights (Hunyuan3D/HF/rembg). Mount this as a volume in Docker to avoid re-downloading.',
    )
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument(
        '--prefetch_models',
        action='store_true',
        help='Download/mirror model files into --model-cache-dir and exit (for faster startup and Docker builds).',
    )
    parser.add_argument(
        '--lazy_load_models',
        action='store_true',
        help='Defer loading large models until first use (slower first request, lower startup RAM/VRAM).',
    )
    parser.add_argument(
        '--idle_unload_sec',
        type=float,
        default=(lambda _v: _v if _v is not None else 0.0)(_env_float('HY3D_IDLE_SECONDS')),
        help='Unload models after N seconds of inactivity (0 disables). You can also set HY3D_IDLE_SECONDS (e.g. via .env).',
    )
    parser.add_argument(
        '--max_vram_gb',
        type=float,
        default=None,
        help='Best-effort VRAM cap (CUDA only). You can also set HY3D_MAX_VRAM_GB.',
    )
    args = parser.parse_args()

    def _configure_model_caches(root_dir: str) -> None:
        root = Path(root_dir).absolute()
        # 1) HY3DGEN internal weights (used by hy3dgen.shapegen/texgen smart loaders)
        os.environ.setdefault('HY3DGEN_MODELS', str(root / 'hy3dgen'))
        # 2) HuggingFace Hub cache (used by diffusers/transformers/controlnets, etc.)
        os.environ.setdefault('HF_HOME', str(root / 'huggingface'))
        os.environ.setdefault('HF_HUB_CACHE', str(root / 'huggingface' / 'hub'))
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(root / 'huggingface' / 'hub'))
        # 3) rembg u2net weights (prevents downloads into /root/.u2net)
        os.environ.setdefault('U2NET_HOME', str(root / 'u2net'))

    _configure_model_caches(args.model_cache_dir)

    def _prefetch_models() -> None:
        from pathlib import Path

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required for --prefetch_models (it should come with diffusers/transformers)."
            ) from e

        def _validate_safetensors_dir(dir_path: Path) -> None:
            # Fail fast if we ended up with a truncated/corrupt safetensors.
            try:
                from safetensors import safe_open
            except Exception:
                # safetensors is part of runtime deps; if missing here, skip validation.
                return

            if not dir_path.exists():
                return

            for p in sorted(dir_path.rglob('*.safetensors')):
                try:
                    if p.stat().st_size <= 0:
                        raise RuntimeError('empty file')
                    with safe_open(str(p), framework='pt', device='cpu') as f:
                        _ = f.metadata()
                        _ = list(f.keys())
                except Exception as e:
                    raise RuntimeError(f"Invalid safetensors under {dir_path}: {p} ({e})")

        hy3dgen_models = Path(os.environ.get('HY3DGEN_MODELS', 'cache/hy3dgen')).expanduser().absolute()
        hy3dgen_models.mkdir(parents=True, exist_ok=True)

        def _snap(repo_id: str, allow_patterns: list[str], local_dir: Path) -> None:
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"[prefetch] repo={repo_id} -> {local_dir}")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=allow_patterns,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            except TypeError:
                # Older huggingface_hub versions may not support resume_download and/or local_dir.
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=allow_patterns,
                        local_dir=str(local_dir),
                        local_dir_use_symlinks=False,
                    )
                except TypeError:
                    snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)

        # ShapeGen (DiT) weights live under HY3DGEN_MODELS/<repo_id>/<subfolder>/...
        _snap(
            args.model_path,
            [f"{args.subfolder}/*"],
            hy3dgen_models / args.model_path,
        )

        _validate_safetensors_dir(hy3dgen_models / args.model_path / args.subfolder)

        # FlashVDM replaces the VAE with a separately-shipped VAE checkpoint.
        if args.enable_flashvdm:
            model_name = str(args.model_path).split('/')[-1]
            turbo_vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini-turbo'),
            }
            if model_name in turbo_vae_mapping:
                vae_repo, vae_subfolder = turbo_vae_mapping[model_name]
                _snap(vae_repo, [f"{vae_subfolder}/*"], hy3dgen_models / vae_repo)
                _validate_safetensors_dir(hy3dgen_models / vae_repo / vae_subfolder)

        # TexGen (Hunyuan3D-2 paint + delight) weights also live under HY3DGEN_MODELS.
        if not args.disable_tex:
            _snap(
                args.texgen_model_path,
                [
                    "hunyuan3d-delight-v2-0/*",
                    "hunyuan3d-paint-v2-0-turbo/*",
                ],
                hy3dgen_models / args.texgen_model_path,
            )

            _validate_safetensors_dir(hy3dgen_models / args.texgen_model_path / 'hunyuan3d-delight-v2-0')
            _validate_safetensors_dir(hy3dgen_models / args.texgen_model_path / 'hunyuan3d-paint-v2-0-turbo')

        # Text-to-image model lives in the HF cache (HF_HOME / HF_HUB_CACHE).
        if args.enable_t23d:
            print("[prefetch] repo=Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled -> HF cache")
            snapshot_download(repo_id='Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')

        # rembg u2net weights (honors U2NET_HOME).
        try:
            _lazy_import_rembg()
            with _ModelLoading():
                _ = BackgroundRemover()
        except Exception as e:
            print(f"[prefetch] rembg prefetch failed (will retry at runtime): {e}")

        print("[prefetch] done")

    if args.prefetch_models:
        _prefetch_models()
        raise SystemExit(0)

    if args.max_vram_gb is None:
        _env_max_vram = os.getenv('HY3D_MAX_VRAM_GB')
        if _env_max_vram:
            try:
                args.max_vram_gb = float(_env_max_vram)
            except ValueError:
                print(f"Invalid HY3D_MAX_VRAM_GB={_env_max_vram!r}; ignoring.")

    def _apply_cuda_vram_cap(max_vram_gb: float | None, device_str: str) -> None:
        # Docker/NVIDIA doesn't provide a strict VRAM limit; this is a best-effort cap
        # for PyTorch's caching allocator.
        if not max_vram_gb or max_vram_gb <= 0:
            return
        if not isinstance(device_str, str) or not device_str.startswith('cuda'):
            return
        if not torch.cuda.is_available():
            return

        device_index = 0
        if ':' in device_str:
            try:
                device_index = int(device_str.split(':', 1)[1])
            except ValueError:
                device_index = 0

        try:
            props = torch.cuda.get_device_properties(device_index)
            total_gb = props.total_memory / (1024 ** 3)
            frac = max_vram_gb / max(total_gb, 1e-6)
            # Keep inside valid range; very tiny fractions tend to break quickly.
            frac = max(0.01, min(1.0, frac))
            torch.cuda.set_per_process_memory_fraction(frac, device=device_index)
            print(
                f"[VRAM cap] device=cuda:{device_index} total={total_gb:.2f}GB "
                f"cap={max_vram_gb:.2f}GB fraction={frac:.4f}"
            )
        except Exception as e:
            print(f"Failed to apply VRAM cap ({max_vram_gb} GB): {e}")

    _apply_cuda_vram_cap(args.max_vram_gb, args.device)

    # Sessions root: must be the same directory we mount as a volume in Docker.
    # Do NOT override --cache-path here; Dockerfile/compose typically passes something like
    # /workspace/outputs, and the Sessions browser should scan that volume.
    if not args.cache_path:
        args.cache_path = 'outputs'
    # Make it absolute to avoid surprises when the process CWD differs.
    SAVE_DIR = str(Path(args.cache_path).expanduser().absolute())
    args.cache_path = SAVE_DIR
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = 'mv' in args.model_path
    TURBO_MODE = 'turbo' in args.subfolder

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()
    example_mvs = get_example_mv_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            # Just a dependency check; actual model load happens on first use.
            _lazy_import_texgen()
            HAS_TEXTUREGEN = True
        except Exception as e:
            print(e)
            print("Failed to import texture generator.")
            print('Please try to install requirements by following README.md')
            HAS_TEXTUREGEN = False

    # Keep the current UX: show the Text Prompt tab, but raise an error if used without --enable_t23d.
    HAS_T2I = True

    if not args.lazy_load_models:
        # Preserve current behavior: load core models on startup.
        get_rmbg_worker()
        get_i23d_worker()
        get_floater_remove_worker()
        get_degenerate_face_remove_worker()
        get_face_reduce_worker()
        if HAS_TEXTUREGEN:
            try:
                get_texgen_worker()
            except Exception as e:
                print(e)
                print("Failed to load texture generator.")
                HAS_TEXTUREGEN = False
        if args.enable_t23d:
            get_t2i_worker()

    if args.idle_unload_sec and args.idle_unload_sec > 0:
        def _idle_unload_loop():
            interval = max(1.0, min(30.0, args.idle_unload_sec / 4.0))
            while True:
                time.sleep(interval)
                now = time.monotonic()
                with _WORKER_LOCK:
                    idle_for = now - _LAST_USE_TS if _LAST_USE_TS else 0.0
                    any_loaded = any(
                        x is not None
                        for x in (
                            i23d_worker,
                            rmbg_worker,
                            texgen_worker,
                            t2i_worker,
                            floater_remove_worker,
                            degenerate_face_remove_worker,
                            face_reduce_worker,
                        )
                    )
                    active = _ACTIVE_CALLS
                    loading = _LOADING_MODELS
                if any_loaded and active == 0 and loading == 0 and idle_for >= args.idle_unload_sec:
                    unload_models(reason=f"idle_for={idle_for:.0f}s")

        threading.Thread(target=_idle_unload_loop, daemon=True).start()

    # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
    # create a FastAPI app
    app = FastAPI()

    @app.get('/model_status')
    async def model_status():
        return _get_model_status_snapshot()

    # create a static directory to store the static files
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)

    if args.low_vram_mode:
        torch.cuda.empty_cache()
    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
