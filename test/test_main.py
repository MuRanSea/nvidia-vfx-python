# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Unit tests for main.py — unified image and video super resolution.

Tests cover helper functions, SuperResProcessor class hierarchy,
and the full pipeline for both image and video modes.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import main as m


# ─────────────────────────────────────────────────────────────────────────────
# Helper function tests (pil_to_rgb_float, rgb_float_to_pil, avframe_to_rgb_float)
# ─────────────────────────────────────────────────────────────────────────────

class TestPilToRgbFloat:
    def test_red_pixel(self):
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        result = m.pil_to_rgb_float(img, gpu=0)

        assert result.shape == (3, 1, 1)
        assert result.dtype == torch.float32
        assert result.device.type == "cuda"

        np.testing.assert_almost_equal(result[0, 0, 0].item(), 1.0, decimal=4)
        np.testing.assert_almost_equal(result[1, 0, 0].item(), 0.0, decimal=4)
        np.testing.assert_almost_equal(result[2, 0, 0].item(), 0.0, decimal=4)

    def test_4x3_green_image(self):
        img = Image.new("RGB", (4, 3), color=(0, 255, 0))
        result = m.pil_to_rgb_float(img, gpu=0)

        assert result.shape == (3, 3, 4)
        assert result.dtype == torch.float32
        assert (result[0] == 0.0).all()
        assert (result[1] == 1.0).all()
        assert (result[2] == 0.0).all()

    def test_black_image(self):
        img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        result = m.pil_to_rgb_float(img, gpu=0)
        assert result.shape == (3, 10, 10)
        assert (result == 0.0).all()

    def test_white_image(self):
        img = Image.new("RGB", (5, 5), color=(255, 255, 255))
        result = m.pil_to_rgb_float(img, gpu=0)
        assert result.shape == (3, 5, 5)
        assert (result == 1.0).all()

    def test_gray_image(self):
        img = Image.new("RGB", (2, 2), color=(128, 128, 128))
        result = m.pil_to_rgb_float(img, gpu=0)
        expected = 128 / 255.0
        np.testing.assert_almost_equal(result[:, 0, 0].cpu().numpy(), [expected] * 3, decimal=4)


class TestRgbFloatToPil:
    def test_black_tensor(self):
        tensor = torch.zeros(3, 10, 10)
        img = m.rgb_float_to_pil(tensor)

        assert isinstance(img, Image.Image)
        assert img.size == (10, 10)
        assert img.mode == "RGB"
        arr = np.array(img)
        assert arr.sum() == 0

    def test_white_tensor(self):
        tensor = torch.ones(3, 5, 5)
        img = m.rgb_float_to_pil(tensor)

        assert isinstance(img, Image.Image)
        assert img.size == (5, 5)
        arr = np.array(img)
        assert arr.sum() == 255 * 5 * 5 * 3

    def test_red_tensor(self):
        tensor = torch.zeros(3, 1, 1)
        tensor[0] = 1.0
        img = m.rgb_float_to_pil(tensor)

        arr = np.array(img)
        assert arr[0, 0, 0] == 255
        assert arr[0, 0, 1] == 0
        assert arr[0, 0, 2] == 0


class TestAvFrameToRgbFloat:
    def test_single_red_pixel(self):
        arr = np.array([[[255, 0, 0]]], dtype=np.uint8)  # (1, 1, 3)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = arr

        result = m.avframe_to_rgb_float(mock_frame, gpu=0)

        assert result.shape == (3, 1, 1)
        assert result.dtype == torch.float32
        np.testing.assert_almost_equal(result[0, 0, 0].item(), 1.0, decimal=4)
        np.testing.assert_almost_equal(result[1, 0, 0].item(), 0.0, decimal=4)
        np.testing.assert_almost_equal(result[2, 0, 0].item(), 0.0, decimal=4)

    def test_4x3_red_image(self):
        arr = np.full((3, 4, 3), [255, 0, 0], dtype=np.uint8)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = arr

        result = m.avframe_to_rgb_float(mock_frame, gpu=0)

        assert result.shape == (3, 3, 4)
        assert (result[0] == 1.0).all()
        assert (result[1] == 0.0).all()
        assert (result[2] == 0.0).all()


class TestRoundtrip:
    """PIL → tensor → PIL roundtrip preserves values within uint8 rounding."""

    def test_pil_tensor_pil_roundtrip(self):
        original = Image.new("RGB", (8, 6), color=(200, 100, 50))
        tensor = m.pil_to_rgb_float(original, gpu=0)
        result = m.rgb_float_to_pil(tensor)

        assert result.size == (8, 6)
        assert result.mode == "RGB"

        result_arr = np.array(result)
        expected_arr = np.array(original)
        np.testing.assert_array_almost_equal(result_arr, expected_arr, decimal=0)


# ─────────────────────────────────────────────────────────────────────────────
# SuperResProcessor — abstract base and class hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TestSuperResProcessorInit:
    def test_image_processor_dimensions(self):
        proc = m.ImageSuperResProcessor(
            gpu=0,
            quality=MagicMock(name="QualityLevel", __members__={}),
            scale=2,
            input_width=640,
            input_height=480,
            output_width=1280,
            output_height=960,
            input_path=Path("in.png"),
            output_path=Path("out.png"),
        )
        assert proc.input_width == 640
        assert proc.input_height == 480
        assert proc.output_width == 1280
        assert proc.output_height == 960
        assert proc.scale == 2
        assert proc.mode == "image"
        assert proc.is_loaded is False

    def test_video_processor_dimensions(self):
        proc = m.VideoSuperResProcessor(
            gpu=0,
            quality=MagicMock(name="QualityLevel", __members__={}),
            scale=3,
            input_width=1920,
            input_height=1080,
            output_width=5760,
            output_height=3240,
            input_path=Path("in.mp4"),
            output_path=Path("out.mp4"),
        )
        assert proc.output_width == 5760
        assert proc.output_height == 3240
        assert proc.mode == "video"


class TestFromPathsImage:
    def test_infers_dimensions_from_image_file(self, monkeypatch, tmp_path):
        # Create a 100x50 test image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (100, 50), color=(0, 0, 0)).save(img_path)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "image"])

        proc = m.SuperResProcessor.from_paths(
            mode="image",
            gpu=0,
            quality=MagicMock(),
            scale=2,
            input_path=img_path,
            output_path=tmp_path / "out.png",
        )

        assert isinstance(proc, m.ImageSuperResProcessor)
        assert proc.input_width == 100
        assert proc.input_height == 50
        assert proc.output_width == 200
        assert proc.output_height == 100

    def test_creates_video_processor_for_video_mode(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "video"])

        mock_stream = MagicMock()
        mock_stream.codec_context.width = 1920
        mock_stream.codec_context.height = 1080
        mock_stream.frames = 100
        mock_stream.average_rate = MagicMock(__float__=lambda _: 30.0)

        mock_container = MagicMock()
        # streams.video[0] → mock_stream
        mock_container.streams.video = MagicMock()
        mock_container.streams.video.__getitem__ = MagicMock(return_value=mock_stream)
        mock_container.close = MagicMock()

        with patch("av.open", return_value=mock_container):
            proc = m.SuperResProcessor.from_paths(
                mode="video",
                gpu=0,
                quality=MagicMock(),
                scale=4,
                input_path=Path("in.mp4"),
                output_path=Path("out.mp4"),
            )

        assert isinstance(proc, m.VideoSuperResProcessor)
        assert proc.input_width == 1920
        assert proc.input_height == 1080
        assert proc.output_width == 7680
        assert proc.output_height == 4320


class TestImageProcessorRun:
    def test_load_sets_is_loaded_true(self, monkeypatch):
        mock_vsr = MagicMock()
        mock_vsr.is_loaded = True
        monkeypatch.setattr("main.VideoSuperRes", lambda device, quality: mock_vsr)
        monkeypatch.setattr("torch.cuda.current_stream", lambda: MagicMock(cuda_stream=0))

        proc = m.ImageSuperResProcessor(
            gpu=0,
            quality=MagicMock(),
            scale=2,
            input_width=100,
            input_height=100,
            output_width=200,
            output_height=200,
            input_path=Path("in.png"),
            output_path=Path("out.png"),
        )
        proc.load()
        assert proc.is_loaded is True

    def test_run_before_load_raises(self, monkeypatch):
        proc = m.ImageSuperResProcessor(
            gpu=0,
            quality=MagicMock(),
            scale=2,
            input_width=100,
            input_height=100,
            output_width=200,
            output_height=200,
            input_path=Path("in.png"),
            output_path=Path("out.png"),
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            proc.run()

    def test_excessive_output_resolution_raises(self, monkeypatch):
        proc = m.ImageSuperResProcessor(
            gpu=0,
            quality=MagicMock(),
            scale=10,  # would produce 10000x10000 > HEVC_MAX
            input_width=1000,
            input_height=1000,
            output_width=10000,
            output_height=10000,
            input_path=Path("in.png"),
            output_path=Path("out.png"),
        )
        proc.load()

        # Validation runs inside run() before _process_and_save()
        with pytest.raises(ValueError, match="exceeds HEVC maximum"):
            proc.run()


# ─────────────────────────────────────────────────────────────────────────────
# parse_args
# ─────────────────────────────────────────────────────────────────────────────

class TestParseArgs:
    def test_defaults_image_mode(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["main.py"])

        args = m.parse_args()
        assert args.mode == "image"
        assert args.scale == 2
        assert args.quality == "HIGH"
        assert "test.png" in args.input

    def test_defaults_video_mode(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "video"])

        args = m.parse_args()
        assert args.mode == "video"
        assert "Drift_RUN_Master_Custom.mp4" in args.input

    def test_custom_args(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            [
                "main.py",
                "--mode", "image",
                "-i", "/custom/in.png",
                "-o", "/custom/out.png",
                "--scale", "4",
                "--quality", "ULTRA",
                "--gpu", "1",
            ],
        )

        args = m.parse_args()
        assert args.input == "/custom/in.png"
        assert args.output == "/custom/out.png"
        assert args.scale == 4
        assert args.quality == "ULTRA"
        assert args.gpu == 1

    def test_invalid_scale(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["main.py", "--scale", "7"])

        with pytest.raises(SystemExit):
            m.parse_args()

    def test_invalid_quality(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["main.py", "--quality", "BAD"])

        with pytest.raises(SystemExit):
            m.parse_args()

    def test_invalid_mode(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "invalid"])

        with pytest.raises(SystemExit):
            m.parse_args()
