# SCNN Lane Detection TensorRT Backend

C++ backend for SCNN lane detection using TensorRT for optimized GPU inference.

## Overview

This package provides a high-performance TensorRT backend for the SCNN (Spatial CNN) lane detection model. This backend is based on the SCNN implementation in `scnn_torch/`.

## Features

- TensorRT-optimized inference with FP16 support
- CUDA-accelerated preprocessing and postprocessing
- Dual output handling (segmentation + lane existence)
- Configurable lane existence threshold
- ROS2/ament_cmake integration

## Prerequisites

- CUDA Toolkit
- TensorRT
- OpenCV
- Trained SCNN checkpoint (from `scnn_torch`)

## Model Export

### Step 1: Export to ONNX

```bash
cd scnn_trt_backend/script
python export_scnn_to_onnx.py \
    --checkpoint ../../scnn_torch/checkpoints/best.pth \
    --height 288 \
    --width 952 \
    --output-dir ../onnxs
```

This generates `onnxs/scnn_vgg16_288x952.onnx`.

### Step 2: Convert to TensorRT Engine

```bash
trtexec --onnx=./onnxs/scnn_vgg16_288x952.onnx \
        --saveEngine=./engines/scnn_vgg16_288x952.engine \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --verbose
```

This generates `engines/scnn_vgg16_288x952.engine`.

**Note:** The engine file is hardware-specific and must be regenerated for different GPUs.

## Building

```bash
colcon build --packages-select scnn_trt_backend
```

The build process will automatically:
1. Export the ONNX model (if checkpoint exists)
2. Generate the TensorRT engine
3. Build the C++ library

## Testing

```bash
colcon test --packages-select scnn_trt_backend --event-handlers console_direct+
```

## Output Format

The `SCNNResult` struct contains:

| Field | Type | Description |
|-------|------|-------------|
| `seg_pred` | `cv::Mat` (CV_8UC3) | Colored lane segmentation mask (BGR) |
| `exist_pred` | `std::array<float, 4>` | Lane existence probabilities [0.0 - 1.0] |

### Lane Colors (BGR)

| Index | Class | Color |
|-------|-------|-------|
| 0 | Background | Black (0, 0, 0) |
| 1 | Lane 1 | Orange (0, 125, 255) |
| 2 | Lane 2 | Green (0, 255, 0) |
| 3 | Lane 3 | Red (0, 0, 255) |
| 4 | Lane 4 | Yellow (0, 255, 255) |

**Note:** Lanes are only colorized if their existence probability exceeds the threshold (default: 0.5).

## Project Structure

```
scnn_trt_backend/
├── cmake/
│   ├── Config.cmake           # Model configuration
│   ├── ModelGeneration.cmake  # ONNX/TensorRT generation
│   └── TestingSetup.cmake     # Test setup
├── engines/
│   └── scnn_vgg16_288x952.engine  # TensorRT engine (generated)
├── include/scnn_trt_backend/
│   ├── config.hpp                 # Constants (normalization, lane colors)
│   ├── decode_and_colorize_kernel.hpp
│   ├── exception.hpp              # Exception classes
│   ├── normalize_kernel.hpp
│   └── scnn_trt_backend.hpp       # Main class declaration
├── onnxs/
│   └── scnn_vgg16_288x952.onnx    # ONNX model (generated)
├── script/
│   ├── export_scnn_to_onnx.py     # ONNX export script
│   └── scnn_lane_detection.py     # Python visualization script
├── src/
│   ├── decode_and_colorize_kernel.cu
│   ├── normalize_kernel.cu
│   └── scnn_trt_backend.cpp       # Main implementation
├── test/
│   ├── test_scnn_trt_backend.cpp
│   ├── image_000.png              # Test image
│   └── image_001.png
├── CMakeLists.txt
├── package.xml
└── ReadMe.md
```

## Configuration

The `Config` struct provides the following options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `height` | 288 | Input image height |
| `width` | 952 | Input image width |
| `num_classes` | 5 | Number of segmentation classes |
| `num_lanes` | 4 | Number of lanes |
| `exist_threshold` | 0.5f | Lane existence threshold |
| `warmup_iterations` | 2 | Engine warmup iterations |
| `log_level` | WARNING | TensorRT log verbosity |
