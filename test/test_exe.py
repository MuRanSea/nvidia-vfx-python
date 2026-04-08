# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
End-to-end test script for the packaged main.exe.

Tests the built standalone executable with both image and video modes.
Run with:  python test_exe.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def exe_path() -> Path:
    """Return path to the packaged main.exe."""
    script_dir = Path(__file__).parent.parent
    exe = script_dir / "build" / "release" / "main.dist" / "main.exe"
    if not exe.exists():
        raise FileNotFoundError(f"main.exe not found at {exe}")
    return exe


def create_test_image(size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Create a simple test image with distinct RGB gradients."""
    w, h = size
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


def test_help(exe: Path) -> None:
    """Test: main.exe --help exits successfully."""
    print("TEST: --help")
    result = subprocess.run(
        [str(exe), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"--help failed: {result.stderr}"
    assert "--mode" in result.stdout, "--mode not in --help output"
    assert "--scale" in result.stdout, "--scale not in --help output"
    print("  PASS: --help")


def test_image_mode(exe: Path, assets_dir: Path, tmp_dir: Path) -> None:
    """Test: main.exe --mode image with assets/input.png."""
    print("TEST: --mode image")
    img_path = assets_dir / "input.png"
    if not img_path.exists():
        print(f"  SKIP: assets/input.png not found at {img_path}")
        return

    out_path = tmp_dir / "test_output.png"

    result = subprocess.run(
        [
            str(exe),
            "--mode", "image",
            "-i", str(img_path),
            "-o", str(out_path),
            "--scale", "2",
            "--quality", "HIGH",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"image mode failed: {result.stderr}\n{result.stdout}"
    assert out_path.exists(), f"Output image not created: {out_path}"

    # Verify output dimensions are 2x of input (256x256 -> 512x512)
    out_img = Image.open(out_path)
    assert out_img.size == (512, 512), f"Expected 512x512, got {out_img.size}"
    print("  PASS: --mode image")


def test_video_mode(exe: Path, assets_dir: Path, tmp_dir: Path) -> None:
    """Test: main.exe --mode video with assets/input.mp4."""
    print("TEST: --mode video")
    video_path = assets_dir / "input.mp4"
    if not video_path.exists():
        print(f"  SKIP: assets/input.mp4 not found at {video_path}")
        return

    out_path = tmp_dir / "test_output.mp4"

    result = subprocess.run(
        [
            str(exe),
            "--mode", "video",
            "-i", str(video_path),
            "-o", str(out_path),
            "--scale", "2",
            "--quality", "HIGH",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"video mode failed: {result.stderr}\n{result.stdout}"
    assert out_path.exists(), f"Output video not created: {out_path}"
    print("  PASS: --mode video")


def test_invalid_mode(exe: Path) -> None:
    """Test: main.exe --mode invalid exits with error."""
    print("TEST: --mode invalid")
    result = subprocess.run(
        [str(exe), "--mode", "invalid"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0, "--mode invalid should have failed"
    print("  PASS: --mode invalid (correctly rejected)")


def test_invalid_scale(exe: Path, tmp_dir: Path) -> None:
    """Test: main.exe --scale 5 exits with error."""
    print("TEST: --scale 5")
    img_path = tmp_dir / "t.png"
    Image.new("RGB", (10, 10)).save(img_path)

    result = subprocess.run(
        [str(exe), "--mode", "image", "-i", str(img_path), "-o", str(tmp_dir / "o.png"), "--scale", "5"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0, "--scale 5 should have failed"
    print("  PASS: --scale 5 (correctly rejected)")


def main() -> None:
    exe = exe_path()
    print(f"Testing: {exe}\n")

    assets_dir = Path(__file__).parent.parent / "assets"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        test_help(exe)
        test_invalid_mode(exe)
        test_invalid_scale(exe, tmp_path)
        test_image_mode(exe, assets_dir, tmp_path)
        test_video_mode(exe, assets_dir, tmp_path)

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
