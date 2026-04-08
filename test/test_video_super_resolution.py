# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Unit tests for video_super_resolution.py

These tests cover parse_args() and avframe_to_rgb_float().
Note: avframe_to_rgb_float requires CUDA to be available.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

import video_super_resolution as vsr


class TestParseArgs:
    def test_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["video_super_resolution.py"])

        args = vsr.parse_args()
        assert args.scale == 2
        assert args.quality == "HIGH"
        # Default paths are computed from __file__ location at module import time,
        # so they point to the project root regardless of cwd
        assert Path(args.input).is_absolute()
        assert Path(args.output).is_absolute()

    def test_custom_args(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            [
                "video_super_resolution.py",
                "-i", "/custom/input.mp4",
                "-o", "/custom/out.mp4",
                "--scale", "4",
                "--quality", "ULTRA",
            ],
        )

        args = vsr.parse_args()
        assert args.input == "/custom/input.mp4"
        assert args.output == "/custom/out.mp4"
        assert args.scale == 4
        assert args.quality == "ULTRA"

    def test_scale_1_denoise(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["video_super_resolution.py", "--scale", "1", "--quality", "DENOISE_HIGH"],
        )

        args = vsr.parse_args()
        assert args.scale == 1
        assert args.quality == "DENOISE_HIGH"

    def test_invalid_scale(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["video_super_resolution.py", "--scale", "5"])

        with pytest.raises(SystemExit):
            vsr.parse_args()

    def test_invalid_quality(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["video_super_resolution.py", "--quality", "INVALID"])

        with pytest.raises(SystemExit):
            vsr.parse_args()


class TestAvFrameToRgbFloat:
    def test_single_red_pixel(self):
        # 1x1 RGB: red=255, green=128, blue=64
        arr = np.array([[[255, 128, 64]]], dtype=np.uint8)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = arr

        result = vsr.avframe_to_rgb_float(mock_frame, gpu=0)

        assert result.shape == (3, 1, 1)
        assert result.dtype == torch.float32

        # R=255/255=1.0, G=128/255≈0.502, B=64/255≈0.251
        np.testing.assert_almost_equal(result[0, 0, 0].item(), 1.0, decimal=4)
        np.testing.assert_almost_equal(result[1, 0, 0].item(), 128 / 255.0, decimal=4)
        np.testing.assert_almost_equal(result[2, 0, 0].item(), 64 / 255.0, decimal=4)

    def test_4x3_image(self):
        # 4x3 RGB image, all pixels red
        arr = np.full((3, 4, 3), [255, 0, 0], dtype=np.uint8)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = arr

        result = vsr.avframe_to_rgb_float(mock_frame, gpu=0)

        assert result.shape == (3, 3, 4)
        assert result.dtype == torch.float32
        assert result.device.type == "cuda"
        # All values should be 1.0 (255/255) since all pixels are red
        assert (result[0] == 1.0).all()
        assert (result[1] == 0.0).all()
        assert (result[2] == 0.0).all()

    def test_black_image(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = arr

        result = vsr.avframe_to_rgb_float(mock_frame, gpu=0)

        assert result.shape == (3, 10, 10)
        assert (result == 0.0).all()

    def test_white_image(self):
        arr = np.full((5, 5, 3), 255, dtype=np.uint8)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = arr

        result = vsr.avframe_to_rgb_float(mock_frame, gpu=0)

        assert result.shape == (3, 5, 5)
        assert (result == 1.0).all()
