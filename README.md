# NVIDIA VFX Python Samples

本仓库包含 NVIDIA Video Effects (VFX) SDK 的 nvidia-vfx Python 绑定的示例应用。Video Super Resolution（视频超分辨率）示例演示了实时视频增强：该应用读取输入视频文件，使用 GPU 加速的 AI 模型对每一帧进行超分辨率处理以提升清晰度并减少压缩伪影，然后根据命令行选项将增强后的结果写入新的输出文件。

## 环境要求

- **Python:** 3.12 或更高版本
- **GPU:** 支持 Tensor Cores 的 NVIDIA GPU（Turing、Ampere、Ada、Blackwell 或 Hopper 架构）
- **GPU 驱动:**
  - Windows: 570.65 或更高版本（TCC 设备需要 595 或更高版本）
  - Linux: 570.190+、580.82+ 或 590.44+
- **操作系统:**
  - Windows: 64 位 Windows 10 或更高版本
  - Linux: Ubuntu 20.04、22.04、24.04、Debian 12 或 RHEL 8/9

## 安装

### 安装依赖

本项目使用 `uv` 管理 Python 依赖，但 `nvidia-vfx` 和 `uitka` 由于加密系统限制无法通过 `uv pip install` 直接安装，需按以下步骤操作：

```bash
uv sync
uv run python -m ensurepip --upgrade
uv run python -m pip install nvidia-vfx==0.1.0.1 --no-cache-dir
uv run python -m pip install -U "Nuitka[all]"
```

## 主入口 — main.py

`main.py` 是本项目的主入口，同时支持**图片超分**和**视频超分**，通过 `--mode` 参数切换。

```bash
# 图片超分（默认）
python main.py --mode image -i input.png -o output.png

# 视频超分
python main.py --mode video -i input.mp4 -o output.mp4

# 常用选项
--scale 4 --quality ULTRA   # 4× 超分辨率，最高音质
--scale 1 --quality DENOISE_HIGH  # 同分辨率降噪，不放大
--gpu 1                     # 指定 GPU 设备
```

### main.py 命令行参数

| 参数               | 说明                                                                                                                                                                                                                                                                                              |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--mode`         | 模式：`image`（默认）或 `video`，决定处理单张图片还是整个视频                                                                                                                                                                                                                                                |
| `-i`, `--input`  | 输入文件路径。默认为 `assets/test.png`（image 模式）或 `assets/Drift_RUN_Master_Custom.mp4`（video 模式）                                                                                                                                                                                                              |
| `-o`, `--output` | 输出文件路径。默认为 `output/test_sr.png`（image 模式）或 `output/sample_sr.mp4`（video 模式）                                                                                                                                                                                                                      |
| `--scale`        | 缩放因子。选项：`1`、`2`、`3`、`4`，默认 `2`。`2`、`3`、`4` 会放大宽高（例如 1920×1080 在 `2`× 下变为 3840×2160），因子越大对 GPU 显存和算力要求越高。设为 `1` 时保持原始分辨率，使用降噪/去模糊模式。                                                                                                                                                             |
| `--quality`      | 处理模式/质量级别。`--scale` ≥ `2` 时可选：`BICUBIC`、`LOW`、`MEDIUM`、`HIGH`（默认）、`ULTRA`，其中 `BICUBIC` 为快速非 AI 基线，`HIGH`/`ULTRA` 最大化细节增强。当 `--scale 1` 时额外可选：`DENOISE_LOW`、`DENOISE_MEDIUM`、`DENOISE_HIGH`、`DENOISE_ULTRA`（去噪/去压缩伪影）以及 `DEBLUR_LOW`、`DEBLUR_MEDIUM`、`DEBLUR_HIGH`、`DEBLUR_ULTRA`（锐化模糊 footage）。 |
| `--gpu`          | CUDA 设备索引，默认 `0`                                                                                                                                                                                                                                                                                         |
| `-h`, `--help`   | 显示帮助信息                                                                                                                                                                                                                                                                                         |

## 相关链接

- [nvidia-vfx Python 包（PyPI）](https://pypi.org/project/nvidia-vfx/)
- [nvidia-vfx Python 绑定使用指南](https://docs.nvidia.com/maxine/vfx-python/latest/index.html)

<br />

