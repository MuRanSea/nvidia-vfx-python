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

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nvvfx import VideoSuperRes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Image Super Resolution using NVIDIA VFX SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(Path(__file__).parent / "assets" / "test.png"),
        help="Input image file (PNG, JPEG, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(Path(__file__).parent / "output" / "test_sr.png"),
        help="Output image file",
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
    return parser.parse_args()


def pil_to_rgb_float(img: Image.Image, gpu: int) -> torch.Tensor:
    arr = np.array(img.convert("RGB"))  # (H, W, 3) uint8
    tensor = torch.from_numpy(arr).to(f"cuda:{gpu}")
    tensor = tensor.permute(2, 0, 1).float() / 255.0  # (3, H, W) float32
    return tensor.contiguous()


def rgb_float_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (
        (tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    )
    return Image.fromarray(arr, mode="RGB")


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quality = VideoSuperRes.QualityLevel[args.quality]
    gpu = 0
    stream_ptr = torch.cuda.current_stream().cuda_stream

    print("=" * 60)
    print("Image Super Resolution")
    print("=" * 60)
    print(f"  Input:   {input_path}")
    print(f"  Output:  {output_path}")
    print(f"  Scale:   {args.scale}x")
    print(f"  Quality: {args.quality}")
    print()

    torch.cuda.set_device(gpu)

    img = Image.open(str(input_path))
    input_width, input_height = img.size
    output_width = input_width * args.scale
    output_height = input_height * args.scale

    print("Image info:")
    print(f"  Resolution: {input_width}x{input_height} -> {output_width}x{output_height}")
    print()

    sr = VideoSuperRes(device=gpu, quality=quality)
    sr.input_width = input_width
    sr.input_height = input_height
    sr.output_width = output_width
    sr.output_height = output_height
    sr.load()
    print(f"Model loaded: {sr.is_loaded}")
    print()

    print("Processing...")
    start_time = time.time()

    rgb_input = pil_to_rgb_float(img, gpu)

    torch.cuda.nvtx.range_push("VideoSuperRes")
    output = sr.run(rgb_input, stream_ptr=stream_ptr)
    rgb_output = torch.from_dlpack(output.image).clone()
    torch.cuda.nvtx.range_pop()

    out_img = rgb_float_to_pil(rgb_output)
    out_img.save(str(output_path))

    elapsed = time.time() - start_time

    print()
    print("Results:")
    print(f"  Time elapsed:     {elapsed:.3f}s")
    output_size = output_path.stat().st_size / 1024
    print(f"  Output size:      {output_size:.2f} KB")
    print(f"  Output file:      {output_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
