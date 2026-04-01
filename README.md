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

### 克隆仓库

```bash
```

初始化 Git LFS 以正确下载示例视频：

```bash
```

### 安装依赖

本项目使用 `uv` 管理 Python 依赖，但 `nvidia-vfx` 由于加密系统限制无法通过 `uv pip install` 直接安装，需按以下步骤操作：

```bash
uv sync
uv run python -m ensurepip --upgrade
uv run python -m pip install nvidia-vfx==0.1.0.1 --no-cache-dir
```

## 运行应用

在仓库根目录下（确保虚拟环境已激活）：

```bash
# 默认：2× 超分辨率，HIGH 质量，使用内置示例视频
python video_super_resolution.py

# 指定输入输出路径
python video_super_resolution.py -i path/to/input.mp4 -o path/to/output.mp4

# 4× 超分辨率，ULTRA 质量
python video_super_resolution.py --scale 4 --quality ULTRA

# 同分辨率降噪（不放大）
python video_super_resolution.py --scale 1 --quality DENOISE_HIGH
```

## 命令行参数说明

| 参数               | 说明                                                                                                                                                                                                                                                                                              |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-i`, `--input`  | 输入视频文件路径。默认为 `assets/Drift_RUN_Master_Custom.mp4`（相对于脚本所在目录）                                                                                                                                                                                                                                    |
| `-o`, `--output` | 输出视频文件路径。默认为 `output/sample_sr.mp4`（相对于脚本所在目录）                                                                                                                                                                                                                                                  |
| `--scale`        | 缩放因子。选项：`1`、`2`、`3`、`4`，默认 `2`。`2`、`3`、`4` 会放大宽高（例如 1920×1080 在 `2`× 下变为 3840×2160），因子越大对 GPU 显存和算力要求越高。设为 `1` 时保持原始分辨率，使用降噪/去模糊模式。                                                                                                                                                             |
| `--quality`      | 处理模式/质量级别。`--scale` ≥ `2` 时可选：`BICUBIC`、`LOW`、`MEDIUM`、`HIGH`（默认）、`ULTRA`，其中 `BICUBIC` 为快速非 AI 基线，`HIGH`/`ULTRA` 最大化细节增强。当 `--scale 1` 时额外可选：`DENOISE_LOW`、`DENOISE_MEDIUM`、`DENOISE_HIGH`、`DENOISE_ULTRA`（去噪/去压缩伪影）以及 `DEBLUR_LOW`、`DEBLUR_MEDIUM`、`DEBLUR_HIGH`、`DEBLUR_ULTRA`（锐化模糊 footage）。 |
| `-h`, `--help`   | 显示帮助信息。                                                                                                                                                                                                                                                                                         |

## 相关链接

- [nvidia-vfx Python 包（PyPI）](https://pypi.org/project/nvidia-vfx/)
- [nvidia-vfx Python 绑定使用指南](https://docs.nvidia.com/maxine/vfx-python/latest/index.html)

<br />

