#pragma once

// C++ standard library includes
#include <array>
#include <string>

// OpenCV includes
#include <opencv2/core.hpp>

// Torch includes
#include <torch/script.h>
#include <torch/torch.h>


namespace scnn_torch_backend
{

/**
 * @brief Result structure for SCNN inference
 */
struct SCNNResult
{
  cv::Mat seg_pred;                   // Segmentation mask (H, W, 3) colored
  std::array<float, 4> exist_pred;    // Lane existence probabilities [lane1, lane2, lane3, lane4]
};

/**
 * @brief SCNN Lane Detection Backend using LibTorch
 */
class SCNNTorchBackend
{
public:
  /**
   * @brief Construct SCNN backend
   * @param model_path Path to TorchScript model (.pt file)
   * @param device Torch device (torch::kCPU or torch::kCUDA)
   */
  SCNNTorchBackend(const std::string & model_path, torch::Device device = torch::kCPU);

  /**
   * @brief Run lane detection inference
   * @param image Input image (BGR format, CV_8UC3)
   * @return SCNNResult containing colored segmentation mask and lane existence probabilities
   */
  SCNNResult detect(const cv::Mat & image);

private:
  /**
   * @brief Apply lane colormap to segmentation mask
   * @param mask Segmentation mask (H, W) with class indices 0-4
   * @param exist_pred Lane existence probabilities
   * @param threshold Only draw lane if probability > threshold
   * @return Colored mask (H, W, 3) in BGR format
   */
  cv::Mat apply_colormap(
    const cv::Mat & mask,
    const std::array<float, 4> & exist_pred,
    float threshold);

  /**
   * @brief Preprocess image for inference
   * @param image Input image (BGR format)
   * @return Preprocessed tensor ready for model input
   */
  torch::Tensor preprocess(const cv::Mat & image);

private:
  torch::jit::script::Module model_;
  torch::Device device_;
};

}  // namespace scnn_torch_backend
