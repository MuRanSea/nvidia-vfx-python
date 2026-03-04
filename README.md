NVIDIA VFX Python Samples
=========================

This repository contains sample applications for the nvidia-vfx Python bindings of the NVIDIA Video Effects (VFX) SDK. The Video Super Resolution sample is a Python reference application that demonstrates real-time video upscaling: it reads an input video file, upscales each frame using GPU-accelerated AI models to improve sharpness and reduce compression artifacts, and writes the enhanced result to a new output file based on command-line options.

Requirements
------------

- **Python:** 3.10 or later
- **GPU:** NVIDIA GPU with Tensor Cores (Turing, Ampere, Ada, Blackwell, or Hopper architecture)
- **GPU driver:**
  - Windows: 570.65 or later (for Tesla Compute Cluster (TCC) devices, 595 or later is required)
  - Linux: 570.190+, 580.82+, or 590.44+
- **Git:** https://git-scm.com/downloads
- **Git LFS:** https://git-lfs.com/ (required for sample video assets)
- **OS:**
  - Windows: 64-bit Windows 10 or later
  - Linux: Ubuntu 20.04, 22.04, 24.04, Debian 12, or RHEL 8/9

Setup
-----

### Clone the repository

```bash
git clone git@github.com:NVIDIA-Maxine/nvidia-vfx-python-samples.git        # Using SSH, or
# git clone https://github.com/NVIDIA-Maxine/nvidia-vfx-python-samples.git  # Using HTTP
cd nvidia-vfx-python-samples
```

Initialize Git LFS so sample videos are downloaded correctly:

```bash
git lfs install
git lfs pull
```

### Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Linux
# .venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
```

Running the application
-----------------------

From the repository root (with your virtual environment activated):

```bash
# Default: 2× upscale, HIGH quality, using bundled sample video
python video_super_resolution.py

# Custom input and output
python video_super_resolution.py -i path/to/input.mp4 -o path/to/output.mp4

# 4× upscale with ULTRA quality
python video_super_resolution.py --scale 4 --quality ULTRA

# Same-resolution denoise (no upscaling)
python video_super_resolution.py --scale 1 --quality DENOISE_HIGH
```

Video Super Resolution Sample — Command-Line Reference
------------------------------------------------------

| Argument            | Description |
|---------------------|-------------|
| `-i`, `--input`     | The input video file path. Default: `assets/Drift_RUN_Master_Custom.mp4` (relative to the script location) |
| `-o`, `--output`    | The output video file path. Default: `output/sample_sr.mp4` (relative to the script location) |
| `--scale`          | The scale factor. Choices: `1`, `2`, `3`, `4`. Default: `2`. Values `2`, `3`, and `4` upscale both width and height (for example, 1920x1080 → 3840x2160 at `2`×), with higher factors demanding more GPU memory and compute. A value of `1` keeps the original resolution and enables same‑resolution cleanup modes (denoise/deblur) instead of upscaling. |
| `--quality`        | The processing mode / quality level. For `--scale` ≥ `2`, choices: `BICUBIC`, `LOW`, `MEDIUM`, `HIGH`, `ULTRA` (default: `HIGH`), where `BICUBIC` is a fast non‑AI baseline, `LOW`/`MEDIUM` favor speed, and `HIGH`/`ULTRA` maximize detail enhancement and artifact reduction. When `--scale 1` is used, additional choices: `DENOISE_LOW`, `DENOISE_MEDIUM`, `DENOISE_HIGH`, `DENOISE_ULTRA` (noise/compression artifact removal) and `DEBLUR_LOW`, `DEBLUR_MEDIUM`, `DEBLUR_HIGH`, `DEBLUR_ULTRA` (sharpening for soft or blurry footage). |
| `-h`, `--help`      | Display help information for the command. |

Links
-----

- [nvidia-vfx Python package on PyPI](https://pypi.org/project/nvidia-vfx/)
- [nvidia-vfx Python bindings guide](https://docs.nvidia.com/maxine/vfx-python/latest/index.html)

> **Note:** This project is currently not accepting contributions.
