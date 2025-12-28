#include <iostream>

#include <opencv2/imgproc.hpp>

#include "scnn_torch_backend/config.hpp"
#include "scnn_torch_backend/scnn_torch_backend.hpp"


namespace scnn_torch_backend
{

SCNNTorchBackend::SCNNTorchBackend(const std::string & model_path, torch::Device device)
: device_(device)
{
  // Load model
  try {
    model_ = torch::jit::load(model_path);
    model_.to(device_);
    model_.eval();

    std::cout << "SCNN model loaded successfully on " << device_ << std::endl;
  } catch (const c10::Error & e) {
    throw std::runtime_error("Error loading the model: " + std::string(e.what()));
  }
}

SCNNResult SCNNTorchBackend::detect(const cv::Mat & image)
{
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }

  // Store original size
  cv::Size original_size(image.cols, image.rows);

  // Resize input to fixed model size (288x952 for KITTI aspect ratio)
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(config::MODEL_WIDTH, config::MODEL_HEIGHT),
             0.0, 0.0, cv::INTER_LINEAR);

  // Preprocess image
  torch::Tensor input_tensor = preprocess(resized_image);

  // Disable gradient computation for inference
  torch::NoGradGuard no_grad;

  // Run inference
  std::vector<torch::jit::IValue> inputs{input_tensor};
  auto output = model_.forward(inputs);

  // Extract outputs from tuple (seg_pred, exist_pred)
  auto tuple = output.toTuple();
  torch::Tensor seg_output = tuple->elements()[0].toTensor();   // [1, 5, H, W]
  torch::Tensor exist_output = tuple->elements()[1].toTensor(); // [1, 4]

  // Get segmentation mask (argmax over class dimension)
  torch::Tensor mask = seg_output.argmax(1).squeeze(0).to(torch::kU8).cpu();  // [H, W]

  // Apply sigmoid to existence logits to get probabilities
  torch::Tensor exist_probs = torch::sigmoid(exist_output).squeeze(0).cpu();  // [4]

  // Extract existence probabilities
  std::array<float, 4> exist_pred;
  auto exist_accessor = exist_probs.accessor<float, 1>();
  for (int i = 0; i < 4; ++i) {
    exist_pred[i] = exist_accessor[i];
  }

  // Convert segmentation tensor to OpenCV Mat
  cv::Mat mask_mat(config::MODEL_HEIGHT, config::MODEL_WIDTH, CV_8UC1, mask.data_ptr<uint8_t>());
  cv::Mat result_mask = mask_mat.clone();

  // Resize mask back to original size first
  cv::Mat resized_mask;
  cv::resize(result_mask, resized_mask, original_size, 0, 0, cv::INTER_NEAREST);

  // Apply colormap at final resolution (filter by existence threshold)
  cv::Mat colored_mask = apply_colormap(resized_mask, exist_pred, config::EXIST_THRESHOLD);

  // Build result
  SCNNResult result;
  result.seg_pred = colored_mask;
  result.exist_pred = exist_pred;

  return result;
}

cv::Mat SCNNTorchBackend::apply_colormap(
  const cv::Mat & mask,
  const std::array<float, 4> & exist_pred,
  float threshold)
{
  cv::Mat colormap(mask.rows, mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int i = 0; i < mask.rows; ++i) {
    for (int j = 0; j < mask.cols; ++j) {
      size_t label = static_cast<size_t>(mask.at<uint8_t>(i, j));
      // Only draw lane if existence probability > threshold
      // label 0 is background, labels 1-4 are lanes
      if (label > 0 && label < config::LANE_COLORMAP.size()) {
        size_t lane_idx = label - 1;  // Convert label (1-4) to index (0-3)
        if (exist_pred[lane_idx] > threshold) {
          colormap.at<cv::Vec3b>(i, j) = cv::Vec3b(
            config::LANE_COLORMAP[label][0],  // B
            config::LANE_COLORMAP[label][1],  // G
            config::LANE_COLORMAP[label][2]   // R
          );
        }
      }
    }
  }

  return colormap;
}

torch::Tensor SCNNTorchBackend::preprocess(const cv::Mat & image)
{
  cv::Mat float_img;
  image.convertTo(float_img, CV_32FC3, 1.0 / 255);

  // Convert BGR to RGB
  cv::cvtColor(float_img, float_img, cv::COLOR_BGR2RGB);

  // Normalize with ImageNet mean/std
  std::vector<cv::Mat> channels(3);
  cv::split(float_img, channels);
  channels[0] = (channels[0] - config::MEAN[0]) / config::STDDEV[0];
  channels[1] = (channels[1] - config::MEAN[1]) / config::STDDEV[1];
  channels[2] = (channels[2] - config::MEAN[2]) / config::STDDEV[2];
  cv::merge(channels, float_img);

  // Convert to torch tensor
  torch::Tensor tensor_image = torch::from_blob(
    float_img.data, {1, image.rows, image.cols, 3}, torch::kFloat32);
  tensor_image = tensor_image.permute({0, 3, 1, 2}).contiguous();  // [B, C, H, W]

  // Move tensor to the appropriate device
  return tensor_image.to(device_);
}

}  // namespace scnn_torch_backend
