# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the NVIDIA VFX Python SDK samples repository. It demonstrates GPU-accelerated Video and Image Super Resolution (VSR) — AI-powered upscaling/denoising using NVIDIA Tensor Cores.

## Commands

```bash
# Install dependencies (nvidia-vfx must be installed separately due to加密 restrictions)
uv sync
uv run python -m ensurepip --upgrade
uv run python -m pip install nvidia-vfx==0.1.0.1 --no-cache-dir

# Run all tests
uv run pytest test/ -v

# Unified entry point (preferred)
python main.py --mode image -i input.png -o output.png --scale 2 --quality HIGH
python main.py --mode video -i input.mp4 -o output.mp4 --scale 2 --quality HIGH

# Legacy individual entry points
python video_super_resolution.py
python image_super_resolution.py
```

## Architecture

Unified entry point: `main.py`

```
SuperResProcessor (ABC)        ← shared VSR init + inference + NVTX wrapping
├── ImageSuperResProcessor     ← PIL → GPU tensor → VSR → PIL save
├── VideoSuperResProcessor     ← PyAV decode → VSR → H.265 encode → mux
└── from_paths(mode, ...)       ← factory: infers dimensions, returns correct subclass
```

Helper functions (`pil_to_rgb_float`, `rgb_float_to_pil`, `avframe_to_rgb_float`) are standalone and importable from `main`.

Legacy files (`video_super_resolution.py`, `image_super_resolution.py`) remain functional and unchanged.

## Requirements

- Python 3.12+
- NVIDIA GPU with Tensor Cores (Turing/Ampere/Ada/Blackwell/Hopper)
- GPU driver: 570.65+ (Windows), 570.190+/580.82+/590.44+ (Linux)
