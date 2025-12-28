# SCNN Lane Detection Torch Backend

C++ backend for SCNN lane detection using LibTorch.

## Preparation

### Clone PyTorch and TorchVision from GitHub
```bash
cd ~/thirdparty/
git clone git@github.com:pytorch/pytorch.git --recurse-submodules
git clone git@github.com:pytorch/vision.git --recurse-submodules
```

### Build LibTorch
```bash
cd ~/thirdparty/pytorch
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DUSE_CUDA=ON \
         -DUSE_CUDNN=ON \
         -DUSE_CUDSS=ON \
         -DUSE_CUFILE=ON \
         -DUSE_CUSPARSELT=ON \
         -DCMAKE_INSTALL_PREFIX=$HOME/thirdparty/libtorch
cmake --build . -j8
cmake --install .
```

If you encountered issues, try adding the following to your `~/.bashrc` before building libtorch:
```bash
export PATH=/usr/local/cuda/bin:/usr/src/tensorrt/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda/include:$CPATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
```

### Point to your custom libtorch installation
Add the following to your `~/.bashrc` file:
```bash
export Torch_DIR="$HOME/thirdparty/libtorch/share/cmake/Torch"
export LD_LIBRARY_PATH="$HOME/thirdparty/libtorch/lib:$LD_LIBRARY_PATH"
```

### Build LibTorchVision
```bash
cd ~/thirdparty/vision
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON \
         -DWITH_CUDA=ON \
         -DCMAKE_PREFIX_PATH=$HOME/thirdparty/libtorch \
         -DCMAKE_INSTALL_PREFIX=$HOME/thirdparty/libtorchvision
cmake --build . -j8
cmake --install .
```

### Point to your custom libtorchvision installation
Add the following to your `~/.bashrc` file:
```bash
export TorchVision_DIR="$HOME/thirdparty/libtorchvision/share/cmake/TorchVision"
export LD_LIBRARY_PATH="$HOME/thirdparty/libtorchvision/lib:$LD_LIBRARY_PATH"
```

## Model Export

Before building, you need to export the trained SCNN model to TorchScript format.

### Prerequisites
- Trained SCNN checkpoint at `scnn_torch/checkpoints/best.pth`

### Export Model
```bash
cd scnn_torch_backend/script
python export_scnn_to_pt.py
```

This will generate `scnn_torch_backend/models/scnn_vgg16_288x800.pt`.

Custom export options:
```bash
python export_scnn_to_pt.py --height 288 --width 800 --checkpoint /path/to/checkpoint.pth --output-dir models
```

## Building

```bash
colcon build --packages-select scnn_torch_backend
```

## Testing

```bash
colcon test --packages-select scnn_torch_backend
```

Make sure to copy test images to `scnn_torch_backend/test/` before running tests.

## Usage

```cpp
#include "scnn_torch_backend/scnn_torch_backend.hpp"

// Initialize detector (CPU)
scnn_torch_backend::SCNNTorchBackend detector("scnn_vgg16_288x800.pt", torch::kCPU);

// Or with CUDA
scnn_torch_backend::SCNNTorchBackend detector("scnn_vgg16_288x800.pt", torch::kCUDA);

// Run inference
cv::Mat image = cv::imread("image.png");
scnn_torch_backend::SCNNResult result = detector.detect(image);

// Access results
cv::Mat lane_mask = result.seg_pred;              // Colored lane segmentation (BGR)
std::array<float, 4> exist = result.exist_pred;   // Lane existence probabilities
```

## Project Structure

```
scnn_torch_backend/
├── cmake/
│   ├── Config.cmake           # Model configuration
│   ├── ModelGeneration.cmake  # TorchScript model generation
│   └── TestingSetup.cmake     # Test setup
├── include/scnn_torch_backend/
│   ├── config.hpp             # Constants (normalization, lane colors)
│   └── scnn_torch_backend.hpp # Main class declaration
├── models/
│   └── scnn_vgg16_288x800.pt        # Exported TorchScript model (generated)
├── script/
│   ├── export_scnn_to_pt.py   # Model export script
│   └── scnn_lane_detection.py # Python visualization script
├── src/
│   └── scnn_torch_backend.cpp # Implementation
├── test/
│   ├── test_scnn_torch_backend.cpp
│   └── image_000.png          # Test image
├── CMakeLists.txt
├── package.xml
└── ReadMe.md
```

## Output Format

The `SCNNResult` struct contains:

| Field | Type | Description |
|-------|------|-------------|
| `seg_pred` | `cv::Mat` (CV_8UC3) | Colored lane segmentation mask (BGR) |
| `exist_pred` | `std::array<float, 4>` | Lane existence probabilities [0.0 - 1.0] |

Lane colors (BGR):
| Index | Lane | Color |
|-------|------|-------|
| 0 | Background | Black (0, 0, 0) |
| 1 | Lane 1 | Orange (0, 125, 255) |
| 2 | Lane 2 | Green (0, 255, 0) |
| 3 | Lane 3 | Red (0, 0, 255) |
| 4 | Lane 4 | Yellow (0, 255, 255) |

## Reference

This backend is based on the SCNN implementation in `scnn_torch/`.

```bibtex
@inproceedings{pan2018spatial,
  title={Spatial as deep: Spatial cnn for traffic scene understanding},
  author={Pan, Xingang and Shi, Jianping and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={AAAI},
  year={2018}
}
```
