#pragma once

// C++ standard library includes
#include <array>


namespace config
{

// Model input size (fixed for traced model)
// 288x952 preserves KITTI aspect ratio (370x1226 -> 288x952)
constexpr int MODEL_HEIGHT = 288;
constexpr int MODEL_WIDTH = 952;

// Lane existence threshold (probability > threshold to draw lane)
constexpr float EXIST_THRESHOLD = 0.5f;

// ImageNet normalization constants
constexpr std::array<float, 3> MEAN = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> STDDEV = {0.229f, 0.224f, 0.225f};

// Lane colors for visualization (BGR format for OpenCV)
// Index 0: Background (black)
// Index 1-4: Lane 1-4
constexpr std::array<std::array<unsigned char, 3>, 5> LANE_COLORMAP = {{
  {0, 0, 0},        // Background: Black
  {0, 125, 255},    // Lane 1: Orange (RGB: 255, 125, 0)
  {0, 255, 0},      // Lane 2: Green  (RGB: 0, 255, 0)
  {0, 0, 255},      // Lane 3: Red    (RGB: 255, 0, 0)
  {0, 255, 255},    // Lane 4: Yellow (RGB: 255, 255, 0)
}};

}  // namespace config
