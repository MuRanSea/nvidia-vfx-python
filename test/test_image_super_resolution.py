# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Unit tests for image_super_resolution.py

Tests cover parse_args(), pil_to_rgb_float(), and rgb_float_to_pil().
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import image_super_resolution as isr


class TestParseArgs:
    def test_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["image_super_resolution.py"])

        args = isr.parse_args()
        assert args.scale == 2
        assert args.quality == "HIGH"
        assert Path(args.input).is_absolute()
        assert Path(args.output).is_absolute()

    def test_custom_args(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            [
                "image_super_resolution.py",
                "-i", "/custom/input.png",
                "-o", "/custom/out.png",
                "--scale", "4",
                "--quality", "ULTRA",
            ],
        )

        args = isr.parse_args()
        assert args.input == "/custom/input.png"
        assert args.output == "/custom/out.png"
        assert args.scale == 4
        assert args.quality == "ULTRA"

    def test_scale_1_denoise(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv",
            ["image_super_resolution.py", "--scale", "1", "--quality", "DENOISE_HIGH"],
        )

        args = isr.parse_args()
        assert args.scale == 1
        assert args.quality == "DENOISE_HIGH"

    def test_invalid_scale(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["image_super_resolution.py", "--scale", "5"])

        with pytest.raises(SystemExit):
            isr.parse_args()

    def test_invalid_quality(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["image_super_resolution.py", "--quality", "INVALID"])

        with pytest.raises(SystemExit):
            isr.parse_args()


class TestPilToRgbFloat:
    def test_red_pixel(self):
        img = Image.new("RGB", (1, 1), color=(255, 0, 0))
        result = isr.pil_to_rgb_float(img, gpu=0)

        assert result.shape == (3, 1, 1)
        assert result.dtype == torch.float32
        assert result.device.type == "cuda"

        np.testing.assert_almost_equal(result[0, 0, 0].item(), 1.0, decimal=4)
        np.testing.assert_almost_equal(result[1, 0, 0].item(), 0.0, decimal=4)
        np.testing.assert_almost_equal(result[2, 0, 0].item(), 0.0, decimal=4)

    def test_4x3_image(self):
        img = Image.new("RGB", (4, 3), color=(0, 255, 0))  # green
        result = isr.pil_to_rgb_float(img, gpu=0)

        assert result.shape == (3, 3, 4)
        assert result.dtype == torch.float32
        # R=0, G=255/255=1.0, B=0
        assert (result[0] == 0.0).all()
        assert (result[1] == 1.0).all()
        assert (result[2] == 0.0).all()

    def test_black_image(self):
        img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        result = isr.pil_to_rgb_float(img, gpu=0)

        assert result.shape == (3, 10, 10)
        assert (result == 0.0).all()

    def test_white_image(self):
        img = Image.new("RGB", (5, 5), color=(255, 255, 255))
        result = isr.pil_to_rgb_float(img, gpu=0)

        assert result.shape == (3, 5, 5)
        assert (result == 1.0).all()

    def test_gray_image(self):
        img = Image.new("RGB", (2, 2), color=(128, 128, 128))
        result = isr.pil_to_rgb_float(img, gpu=0)

        expected = 128 / 255.0
        np.testing.assert_almost_equal(result[0, 0, 0].item(), expected, decimal=4)
        np.testing.assert_almost_equal(result[1, 0, 0].item(), expected, decimal=4)
        np.testing.assert_almost_equal(result[2, 0, 0].item(), expected, decimal=4)


class TestRgbFloatToPil:
    def test_black_tensor(self):
        tensor = torch.zeros(3, 10, 10)
        img = isr.rgb_float_to_pil(tensor)

        assert isinstance(img, Image.Image)
        assert img.size == (10, 10)
        assert img.mode == "RGB"
        arr = np.array(img)
        assert arr.sum() == 0

    def test_white_tensor(self):
        tensor = torch.ones(3, 5, 5)
        img = isr.rgb_float_to_pil(tensor)

        assert isinstance(img, Image.Image)
        assert img.size == (5, 5)
        arr = np.array(img)
        assert arr.sum() == 255 * 5 * 5 * 3

    def test_red_tensor(self):
        tensor = torch.zeros(3, 1, 1)
        tensor[0] = 1.0  # R channel
        img = isr.rgb_float_to_pil(tensor)

        arr = np.array(img)
        assert arr[0, 0, 0] == 255
        assert arr[0, 0, 1] == 0
        assert arr[0, 0, 2] == 0


class TestRoundtrip:
    def test_pil_tensor_pil_roundtrip(self):
        original = Image.new("RGB", (8, 6), color=(200, 100, 50))
        tensor = isr.pil_to_rgb_float(original, gpu=0)
        result = isr.rgb_float_to_pil(tensor)

        assert result.size == (8, 6)
        assert result.mode == "RGB"

        # Values should be approximately preserved (within uint8 rounding)
        result_arr = np.array(result)
        expected_arr = np.array(original)
        np.testing.assert_array_almost_equal(result_arr, expected_arr, decimal=0)
