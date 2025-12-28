// C++ standard library includes
#include <chrono>
#include <numeric>
#include <stdexcept>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Google Test includes
#include <gtest/gtest.h>

// Torch includes
#include <torch/torch.h>

// Local includes
#define private public
#include "scnn_torch_backend/scnn_torch_backend.hpp"
#undef private


class SCNNTorchBackendTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // This will be overridden by individual test cases
  }

  void TearDown() override
  {
    // Clean up if needed
  }

  void init_detector(torch::Device device)
  {
    try {
      detector_ = std::make_unique<scnn_torch_backend::SCNNTorchBackend>(model_path_, device);
    } catch (const std::exception & e) {
      GTEST_SKIP() << "Failed to initialize SCNN detector: " << e.what();
    }
  }

  cv::Mat load_test_image()
  {
    cv::Mat image = cv::imread(image_path_);
    if (image.empty()) {
      throw std::runtime_error("Failed to load test image: " + image_path_);
    }
    return image;
  }

  cv::Mat create_overlay(const cv::Mat & original, const cv::Mat & segmentation, float alpha = 0.5f)
  {
    cv::Mat overlay;
    cv::addWeighted(original, 1.0f - alpha, segmentation, alpha, 0.0, overlay);
    return overlay;
  }

  void save_results(
    const cv::Mat & original, const cv::Mat & segmentation,
    const cv::Mat & overlay, const std::string & suffix = "")
  {
    cv::imwrite("test_output_original" + suffix + ".png", original);
    cv::imwrite("test_output_segmentation" + suffix + ".png", segmentation);
    cv::imwrite("test_output_overlay" + suffix + ".png", overlay);
  }

  void print_exist_pred(const std::array<float, 4> & exist_pred)
  {
    std::cout << "Lane existence probabilities: [";
    for (size_t i = 0; i < exist_pred.size(); ++i) {
      std::cout << exist_pred[i];
      if (i < exist_pred.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  std::unique_ptr<scnn_torch_backend::SCNNTorchBackend> detector_;

private:
  const std::string model_path_ = "scnn_vgg16_288x952.pt";
  const std::string image_path_ = "image_000.png";
};


TEST_F(SCNNTorchBackendTest, TestBasicInferenceCPU)
{
  torch::Device device = torch::kCPU;
  init_detector(device);

  cv::Mat image = load_test_image();

  // Validate input image
  EXPECT_FALSE(image.empty());
  EXPECT_EQ(image.type(), CV_8UC3);
  EXPECT_GT(image.rows, 0);
  EXPECT_GT(image.cols, 0);

  std::cout << "Input image size: " << image.rows << "x" << image.cols << std::endl;
  std::cout << "Using device: " << device << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  scnn_torch_backend::SCNNResult result = detector_->detect(image);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

  // Validate segmentation output
  EXPECT_FALSE(result.seg_pred.empty());
  EXPECT_EQ(result.seg_pred.rows, image.rows);
  EXPECT_EQ(result.seg_pred.cols, image.cols);
  EXPECT_EQ(result.seg_pred.type(), CV_8UC3);

  // Validate existence output
  print_exist_pred(result.exist_pred);
  for (size_t i = 0; i < result.exist_pred.size(); ++i) {
    EXPECT_GE(result.exist_pred[i], 0.0f) << "Existence probability should be >= 0";
    EXPECT_LE(result.exist_pred[i], 1.0f) << "Existence probability should be <= 1";
  }

  // Create overlay
  cv::Mat overlay = create_overlay(image, result.seg_pred, 0.5f);
  EXPECT_EQ(overlay.size(), image.size());
  EXPECT_EQ(overlay.type(), CV_8UC3);

  // Save results for visual inspection
  save_results(image, result.seg_pred, overlay, "_cpu");
}


TEST_F(SCNNTorchBackendTest, TestBasicInferenceCUDA)
{
  torch::Device device = torch::kCUDA;
  init_detector(device);

  cv::Mat image = load_test_image();

  // Validate input image
  EXPECT_FALSE(image.empty());
  EXPECT_EQ(image.type(), CV_8UC3);
  EXPECT_GT(image.rows, 0);
  EXPECT_GT(image.cols, 0);

  std::cout << "Input image size: " << image.rows << "x" << image.cols << std::endl;
  std::cout << "Using device: " << device << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  scnn_torch_backend::SCNNResult result = detector_->detect(image);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(end - start);
  std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

  // Validate segmentation output
  EXPECT_FALSE(result.seg_pred.empty());
  EXPECT_EQ(result.seg_pred.rows, image.rows);
  EXPECT_EQ(result.seg_pred.cols, image.cols);
  EXPECT_EQ(result.seg_pred.type(), CV_8UC3);

  // Validate existence output
  print_exist_pred(result.exist_pred);
  for (size_t i = 0; i < result.exist_pred.size(); ++i) {
    EXPECT_GE(result.exist_pred[i], 0.0f) << "Existence probability should be >= 0";
    EXPECT_LE(result.exist_pred[i], 1.0f) << "Existence probability should be <= 1";
  }

  // Create overlay
  cv::Mat overlay = create_overlay(image, result.seg_pred, 0.5f);
  EXPECT_EQ(overlay.size(), image.size());
  EXPECT_EQ(overlay.type(), CV_8UC3);

  // Save results for visual inspection
  save_results(image, result.seg_pred, overlay, "_gpu");
}
