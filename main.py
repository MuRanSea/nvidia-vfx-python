# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
Unified Image and Video Super Resolution using NVIDIA VFX SDK.

Usage:
    python main.py --mode image -i input.png -o output.png --scale 2 --quality HIGH
    python main.py --mode video -i input.mp4 -o output.mp4 --scale 2 --quality HIGH
"""

from __future__ import annotations

import argparse
import sys
import time
from abc import ABC, abstractmethod
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from typing_extensions import Self

import av

from nvvfx import VideoSuperRes


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (pure, no VSR dependency)
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_rgb_float(img: Image.Image, gpu: int) -> torch.Tensor:
    """Convert a PIL RGB image to a (3, H, W) float32 CUDA tensor."""
    arr = np.array(img.convert("RGB"))  # (H, W, 3) uint8
    tensor = torch.from_numpy(arr).to(f"cuda:{gpu}")
    tensor = tensor.permute(2, 0, 1).float() / 255.0  # (3, H, W) float32
    return tensor.contiguous()


def rgb_float_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a (3, H, W) float32 CUDA tensor to a PIL RGB image."""
    arr = (
        (tensor.clamp(0.0, 1.0) * 255.0)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    return Image.fromarray(arr, mode="RGB")


def avframe_to_rgb_float(frame: av.VideoFrame, gpu: int) -> torch.Tensor:
    """Convert an av.VideoFrame to a (3, H, W) float32 CUDA tensor."""
    arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
    tensor = torch.from_numpy(arr).to(f"cuda:{gpu}")
    tensor = tensor.permute(2, 0, 1).float() / 255.0  # (3, H, W) float32
    return tensor.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# SuperResProcessor — abstract base
# ─────────────────────────────────────────────────────────────────────────────

class SuperResProcessor(ABC):
    """
    Abstract base for GPU-accelerated super-resolution processors.

    Subclasses implement `_process_input()` and `_save_output()` to handle
    their specific input/output formats while sharing the VSR inference logic.

    Parameters
    ----------
    gpu : int
        CUDA device index.
    quality : VideoSuperRes.QualityLevel
        Quality/denoise level for the VSR model.
    scale : int
        Upscale factor (1 = same-resolution denoise/deblur, 2/3/4 = upscale).
    input_width, input_height : int
        Dimensions of the input content.
    output_width, output_height : int
        Dimensions of the output content (typically input * scale).
    """

    HEVC_MAX = 8192

    def __init__(
        self,
        gpu: int,
        quality: VideoSuperRes.QualityLevel,
        scale: int,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        self.gpu = gpu
        self.quality = quality
        self.scale = scale
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height
        self._sr: VideoSuperRes | None = None
        self._stream_ptr: int | None = None

    @classmethod
    def from_paths(
        cls,
        mode: str,
        gpu: int,
        quality: VideoSuperRes.QualityLevel,
        scale: int,
        input_path: Path,
        output_path: Path,
    ) -> Self:
        """Factory: infer dimensions from the input file, then construct."""
        if mode == "image":
            img = Image.open(str(input_path))
            w, h = img.size
            return ImageSuperResProcessor(
                gpu=gpu,
                quality=quality,
                scale=scale,
                input_width=w,
                input_height=h,
                output_width=w * scale,
                output_height=h * scale,
                input_path=input_path,
                output_path=output_path,
            )
        elif mode == "video":
            container = av.open(str(input_path))
            stream = container.streams.video[0]
            w = stream.codec_context.width
            h = stream.codec_context.height
            container.close()
            return VideoSuperResProcessor(
                gpu=gpu,
                quality=quality,
                scale=scale,
                input_width=w,
                input_height=h,
                output_width=w * scale,
                output_height=h * scale,
                input_path=input_path,
                output_path=output_path,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True after load() has been called and the model is ready."""
        return self._sr is not None and self._sr.is_loaded

    def load(self) -> None:
        """Initialize and load the VSR model onto the GPU."""
        torch.cuda.set_device(self.gpu)
        self._stream_ptr = torch.cuda.current_stream().cuda_stream

        self._sr = VideoSuperRes(device=self.gpu, quality=self.quality)
        self._sr.input_width = self.input_width
        self._sr.input_height = self.input_height
        self._sr.output_width = self.output_width
        self._sr.output_height = self.output_height
        self._sr.load()

    def run(self) -> None:
        """
        Process the input and write the output.
        Subclasses override this to add loop/progress logic.
        """
        if not self.is_loaded:
            raise RuntimeError("Processor not loaded — call load() first")
        self._validate_output_resolution()
        self._process_and_save()

    # ── internal helpers ─────────────────────────────────────────────────────

    def _run_inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Run VSR inference on a (3, H, W) float32 CUDA tensor.

        Wraps the call in NVTX ranges for GPU profiling.
        Returns the output as a plain CUDA tensor (clone from DLPack).
        """
        torch.cuda.nvtx.range_push("VideoSuperRes")
        output = self._sr.run(tensor, stream_ptr=self._stream_ptr)
        result = torch.from_dlpack(output.image).clone()
        torch.cuda.nvtx.range_pop()
        return result

    def _validate_output_resolution(self) -> None:
        if self.output_height > self.HEVC_MAX or self.output_width > self.HEVC_MAX:
            raise ValueError(
                f"Output resolution {self.output_width}x{self.output_height} "
                f"exceeds HEVC maximum {self.HEVC_MAX}x{self.HEVC_MAX}"
            )

    # ── abstract methods ────────────────────────────────────────────────────

    @abstractmethod
    def _process_and_save(self) -> None:
        """Read input, run inference, write output. Called from run()."""
        ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """Return 'image' or 'video'."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# ImageSuperResProcessor
# ─────────────────────────────────────────────────────────────────────────────

class ImageSuperResProcessor(SuperResProcessor):
    """
    Super-resolution for a single image file (PNG, JPEG, etc.).

    Input  → PIL.Image → RGB float tensor → VSR → PIL.Image → saved to disk
    """

    def __init__(
        self,
        gpu: int,
        quality: VideoSuperRes.QualityLevel,
        scale: int,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        input_path: Path,
        output_path: Path,
    ) -> None:
        super().__init__(gpu, quality, scale, input_width, input_height, output_width, output_height)
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    @property
    def mode(self) -> str:
        return "image"

    def run(self) -> None:
        print("=" * 60)
        print("Image Super Resolution")
        print("=" * 60)
        self._print_info()
        print()
        super().run()

    def _print_info(self) -> None:
        print(f"  Input:   {self.input_path}")
        print(f"  Output:  {self.output_path}")
        print(f"  Scale:   {self.scale}x")
        print(f"  Quality: {self.quality.name}")
        print()
        print("Image info:")
        print(
            f"  Resolution: {self.input_width}x{self.input_height} "
            f"-> {self.output_width}x{self.output_height}"
        )

    def _process_and_save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(str(self.input_path))
        rgb_input = pil_to_rgb_float(img, gpu=self.gpu)

        start_time = time.time()
        rgb_output = self._run_inference(rgb_input)
        elapsed = time.time() - start_time

        out_img = rgb_float_to_pil(rgb_output)
        out_img.save(str(self.output_path))

        self._print_results(elapsed)

    def _print_results(self, elapsed: float) -> None:
        print()
        print("Results:")
        print(f"  Time elapsed: {elapsed:.3f}s")
        output_size = self.output_path.stat().st_size / 1024
        print(f"  Output size:  {output_size:.2f} KB")
        print(f"  Output file:  {self.output_path}")
        print()
        print("Done!")


# ─────────────────────────────────────────────────────────────────────────────
# VideoSuperResProcessor
# ─────────────────────────────────────────────────────────────────────────────

class VideoSuperResProcessor(SuperResProcessor):
    """
    Super-resolution for a video file.

    Input  → av.VideoFrame → RGB float tensor → VSR → H.265 encode → muxed to file
    """

    def __init__(
        self,
        gpu: int,
        quality: VideoSuperRes.QualityLevel,
        scale: int,
        input_width: int,
        input_height: int,
        output_width: int,
        output_height: int,
        input_path: Path,
        output_path: Path,
        bitrate: int = 16_000_000,
    ) -> None:
        super().__init__(gpu, quality, scale, input_width, input_height, output_width, output_height)
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.bitrate = bitrate

    @property
    def mode(self) -> str:
        return "video"

    def run(self) -> None:
        print("=" * 60)
        print("Video Super Resolution")
        print("=" * 60)
        self._print_info()
        print()
        super().run()

    def _print_info(self) -> None:
        print(f"  Input:   {self.input_path}")
        print(f"  Output:  {self.output_path}")
        print(f"  Scale:   {self.scale}x")
        print(f"  Quality: {self.quality.name}")
        print()
        print("Video info:")
        print(
            f"  Resolution: {self.input_width}x{self.input_height} "
            f"-> {self.output_width}x{self.output_height}"
        )
        container = av.open(str(self.input_path))
        stream = container.streams.video[0]
        total_frames = stream.frames or 0
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if total_frames:
            print(f"  Frames:     {total_frames}")
        print(f"  FPS:        {fps:.2f}")
        container.close()
        print()

    def _process_and_save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        input_container = av.open(str(self.input_path))
        input_stream = input_container.streams.video[0]
        input_stream.thread_type = "AUTO"

        total_frames = input_stream.frames or 0
        fps = float(input_stream.average_rate) if input_stream.average_rate else 0.0
        frame_rate = Fraction(fps if fps else 30).limit_denominator(10000)

        # Pick codec: prefer HW (hevc_nvenc), fall back to SW (libx265)
        codec_candidates = ("hevc_nvenc", "libx265")
        output_container = None
        video_stream = None
        codec_name = None
        for name in codec_candidates:
            container = av.open(str(self.output_path), mode="w")
            try:
                stream = container.add_stream(name, rate=frame_rate)
                stream.width = self.output_width
                stream.height = self.output_height
                stream.pix_fmt = "yuv420p"
                stream.bit_rate = self.bitrate
                stream.codec_context.open()
            except Exception:
                container.close()
                continue
            output_container, video_stream, codec_name = container, stream, name
            break

        if video_stream is None:
            raise RuntimeError(f"No usable H.265 encoder (tried {codec_candidates})")

        print(f"Encoder: {codec_name}")

        if total_frames:
            print(f"Processing {total_frames} frames...")
        else:
            print("Processing frames...")
        start_time = time.time()

        processed = 0
        for frame in input_container.decode(input_stream):
            rgb_input = avframe_to_rgb_float(frame, gpu=self.gpu)
            rgb_output = self._run_inference(rgb_input)

            frame_np = (
                (rgb_output.clamp(0.0, 1.0) * 255.0)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            out_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in video_stream.encode(out_frame):
                output_container.mux(packet)
            processed += 1

        for packet in video_stream.encode(None):
            output_container.mux(packet)
        output_container.close()
        input_container.close()

        elapsed = time.time() - start_time
        fps_proc = processed / elapsed if elapsed > 0 else 0

        print()
        print("Results:")
        print(f"  Frames processed: {processed}")
        print(f"  Time elapsed:     {elapsed:.1f}s")
        print(f"  Processing FPS:   {fps_proc:.1f}")
        output_size = self.output_path.stat().st_size / 1024 / 1024
        print(f"  Output size:      {output_size:.2f} MB")
        print(f"  Output file:      {self.output_path}")
        print()
        print("Done!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image / Video Super Resolution using NVIDIA VFX SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video"],
        default="image",
        help="'image' or 'video'",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input file (image or video). Defaults depend on --mode.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file. Defaults depend on --mode.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=[1, 2, 3, 4],
        default=2,
        help="Scale factor: 1 = same-resolution (denoise/deblur), 2/3/4 = upscale",
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=VideoSuperRes.QualityLevel.__members__.keys(),
        default="HIGH",
        help="Super resolution quality level",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index",
    )
    args = parser.parse_args()

    # Resolve default paths relative to this script's location
    script_dir = Path(__file__).parent
    if args.input is None:
        args.input = str(
            script_dir / ("assets/test.png" if args.mode == "image" else "assets/Drift_RUN_Master_Custom.mp4")
        )
    if args.output is None:
        args.output = str(script_dir / ("output/test_sr.png" if args.mode == "image" else "output/sample_sr.mp4"))

    return args


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    quality = VideoSuperRes.QualityLevel[args.quality]

    processor = SuperResProcessor.from_paths(
        mode=args.mode,
        gpu=args.gpu,
        quality=quality,
        scale=args.scale,
        input_path=input_path,
        output_path=output_path,
    )

    processor.load()
    print(f"Model loaded: {processor.is_loaded}")
    print()
    processor.run()


if __name__ == "__main__":
    main()
