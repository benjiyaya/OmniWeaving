# OmniWeaving Windows 11 Setup Guide

This guide documents the complete process for running the [OmniWeaving](https://github.com/Tencent-Hunyuan/OmniWeaving) video generation repo on Windows 11 with a single GPU.

## Prerequisites

- **Python 3.12** (tested with 3.12.10)
- **NVIDIA GPU** with CUDA drivers installed (tested with CUDA 13.0, RTX Pro 6000 Blackwell 96GB)
- **Git** installed

## Step 1: Create Virtual Environment

```cmd
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
```

## Step 2: Install Triton for Windows

Standard `triton` is Linux-only. Use the Windows port instead:

```cmd
pip install -U "triton-windows<3.5"
```

## Step 3: Install CUDA PyTorch

Install PyTorch with CUDA support. The repo pins `torchaudio==2.6.0`, so match the PyTorch version:

```cmd
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

**Note:** If you prefer a newer PyTorch (e.g., 2.9.0), also update `torchaudio` to match:
```cmd
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

## Step 4: Install Remaining Dependencies

```cmd
pip install tqdm==4.67.1 peft==0.17.0 openai==2.8.0 einops==0.8.0 loguru==0.7.3 numpy==1.26.4 imageio==2.37.0 imageio-ffmpeg==0.6.0 "omegaconf>=2.3.0" diffusers==0.32.2 "safetensors>=0.5.3" qwen-vl-utils==0.0.8 huggingface-hub==0.34.0 "huggingface_hub[cli]" "transformers[accelerate,tiktoken]==4.57.1" pillow wheel psutil setuptools modelscope threadpoolctl datasets pyarrow shortuuid decord gradio
```

Then install `angelslim` without deps (it requires Linux-only `triton`):
```cmd
pip install angelslim==0.2.1 --no-deps
```

**Downgrade numpy** if torch installed a newer version:
```cmd
pip install numpy==1.26.4
```

## Step 5: Download Model Weights

```cmd
huggingface-cli download tencent/HY-OmniWeaving --local-dir ./ckpts
```

## Step 6: Apply Windows Compatibility Patches

### Problem 1: NCCL is Linux-only

The repo uses `torch.distributed` with NCCL even for single-GPU inference. NCCL is not available on Windows.

**Fix:** Modify `hyvideo/commons/parallel_states.py` to add a Windows single-GPU fallback.

1. Add import and helper function at the top:
```python
import sys

def _is_windows_single_gpu(world_size):
    return sys.platform == "win32" and world_size == 1
```

2. In `build_mesh()`, add early return before `init_device_mesh()`:
```python
def build_mesh(self, device_type):
    # ... existing assertions ...

    if _is_windows_single_gpu(self.world_size):
        self.world_mesh = None
        self.fsdp_mesh = None
        self.sp_rank = 0
        self.sp_group = None
        return

    # ... rest of the method (init_device_mesh calls) ...
```

### Problem 2: VAE tile parallel decode uses `dist.all_gather`

The VAE's `tile_parallel_spatial_tiled_decode` calls `dist.all_gather` which fails without an initialized process group.

**Fix:** Modify `hyvideo/models/autoencoders/hunyuanvideo_15_vae.py` around line 901-912:

```python
# Replace:
tiles_gather_list = [torch.empty_like(decoded_tiles) for _ in range(world_size)]
metas_gather_list = [torch.empty_like(decoded_metas) for _ in range(world_size)]
dist.all_gather(tiles_gather_list, decoded_tiles, group=get_parallel_state().sp_group)
dist.all_gather(metas_gather_list, decoded_metas, group=get_parallel_state().sp_group)

# With:
sp_group = get_parallel_state().sp_group
if sp_group is None:
    tiles_gather_list = [decoded_tiles]
    metas_gather_list = [decoded_metas]
else:
    tiles_gather_list = [torch.empty_like(decoded_tiles) for _ in range(world_size)]
    metas_gather_list = [torch.empty_like(decoded_metas) for _ in range(world_size)]
    dist.all_gather(tiles_gather_list, decoded_tiles, group=sp_group)
    dist.all_gather(metas_gather_list, decoded_metas, group=sp_group)
```

## Step 7: Run Inference

You have **two options** to run inference:

---

### Option A: Gradio Web UI (Recommended)

A full-featured web interface with all generation options, dynamic inputs based on task selection, and live progress streaming.

```cmd
.\venv\Scripts\Activate
python gradio_app.py
```

Then open your browser to **http://localhost:7777**.

**Features:**
- All 6 task types with dynamic media inputs
- All generation settings exposed in a clean UI
- Live generation log streaming
- Video preview and download
- Defaults: SageAttention, torch compile, 81 frames, 30 steps

---

### Option B: Command Line

**Important:** Set environment variables to bypass `torch.distributed` initialization:

#### Quick Start (Command Prompt)
```cmd
.\venv\Scripts\Activate

set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1
python generate.py --task t2v --prompt "A cat walking on a beach at sunset" --model_path ./ckpts --output_path ./outputs/t2v.mp4
```

#### Quick Start (PowerShell)
```powershell
.\venv\Scripts\Activate

$env:RANK=0
$env:LOCAL_RANK=0
$env:WORLD_SIZE=1
python generate.py --task t2v --prompt "A cat walking on a beach at sunset" --model_path ./ckpts --output_path ./outputs/t2v.mp4
```

#### Full Command with Options (Recommended)
```cmd
.\venv\Scripts\Activate

set RANK=0
set LOCAL_RANK=0
set WORLD_SIZE=1
python generate.py --task t2v --prompt "A cat walking on a beach at sunset" --model_path ./ckpts --output_path ./outputs/t2v.mp4 --use_sageattn --enable_torch_compile --seed 42 --dtype bf16 --offloading 1 --video_length 81 --num_inference_steps 30
```

## Performance Notes

- **SageAttention** is recommended over SDPA fallback for faster inference
- **torch compile** (`--enable_torch_compile`) speeds up subsequent steps after warmup
- **Expected performance** (RTX Pro 6000 Blackwell, 848x480, 32 frames, 20 steps):
  - First step (warmup): ~47s
  - Subsequent steps: ~3.4s/step
  - Total: ~1:08

## Common Issues

### `torch.distributed.DistStoreError: use_libuv was requested but PyTorch was built without libuv support`
- **Cause:** Windows torchrun bug
- **Fix:** Don't use `torchrun`. Use `python generate.py` with `RANK=0`, `LOCAL_RANK=0`, `WORLD_SIZE=1` env vars instead.

### `RuntimeError: Distributed package doesn't have NCCL built in`
- **Cause:** NCCL is Linux-only
- **Fix:** Apply the `parallel_states.py` patch from Step 6.

### `ValueError: Default process group has not been initialized`
- **Cause:** VAE tile decode tries to use `dist.all_gather` without a process group
- **Fix:** Apply the `hunyuanvideo_15_vae.py` patch from Step 6.

### `ModuleNotFoundError: No module named 'triton'`
- **Cause:** `angelslim` requires `triton` which is Linux-only
- **Fix:** Install `triton-windows` first, then `pip install angelslim==0.2.1 --no-deps`

### Out of Memory
- Try reducing `--video_length` (default 81, try 32 for testing)
- Try reducing `--num_inference_steps` (default 50, try 20 for testing)
- Ensure `--offloading 1` is set
- Set `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`

## Supported Tasks

| Task Flag | Description | Required Inputs |
|-----------|-------------|-----------------|
| `t2v` | Text-to-Video | `--prompt` |
| `i2v` | First-Frame-to-Video | `--prompt`, `--ref_image_paths` |
| `interpolation` | Key-Frames-to-Video | `--prompt`, `--ref_image_paths` (2+ images) |
| `reference2v` | Reference-to-Video | `--prompt`, `--ref_image_paths` (1-4 images) |
| `editing` | Video-to-Video Editing | `--prompt`, `--condition_video_paths` |
| `tiv2v` | Text-Image-Video-to-Video | `--prompt`, `--condition_video_paths`, `--ref_image_paths` |
