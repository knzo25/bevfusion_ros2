// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/lidar_bevfusion/bevfusion_config.hpp"
#include "autoware/lidar_bevfusion/preprocess/precomputed_features.hpp"
#include "autoware/lidar_bevfusion/preprocess/preprocess_kernel.hpp"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include <NvInfer.h>
#include <memory.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

class CustomLogger : public nvinfer1::ILogger
{
  void log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept override
  {
    // suppress info-level messages
    if (severity <= nvinfer1::ILogger::Severity::kINFO) {
      std::cout << msg << std::endl;
    }
  }
};

struct InferDeleter
{
  template <typename T>
  void operator()(T * obj) const
  {
    delete obj;
  }
};

template <typename T>
std::vector<T> load_tensor(const std::string & file_name)
{
  std::ifstream input_file(file_name);
  if (!input_file) {
    std::cerr << "Unable to open file\n";
    return std::vector<T>{};
  }

  std::vector<T> data;
  std::string line;

  // Read the file line-by-line
  while (std::getline(input_file, line)) {
    std::istringstream line_stream(line);
    float value;

    while (line_stream >> value) {
      data.push_back(value);
    }
  }

  input_file.close();  // Close the file

  return data;
}

int main(int argc, char ** argv)
{
  using autoware::lidar_bevfusion::Matrix4fRowM;
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <plugin_library_path> <engine_file_path>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string const plugin_library_path{argv[1]};
  std::string const engine_file_path{argv[2]};

  std::cout << "Plugin library path: " << plugin_library_path << std::endl;
  std::cout << "Engine file path: " << engine_file_path << std::endl;
  std::cout << "Loading engine..." << std::endl;

  CustomLogger logger{};

  // Create CUDA stream.
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  nvinfer1::TensorFormat const expected_format{nvinfer1::TensorFormat::kLINEAR};

  // Deserialize the engine.
  std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime{nvinfer1::createInferRuntime(logger)};
  if (runtime == nullptr) {
    std::cerr << "Failed to create the runtime." << std::endl;
    return EXIT_FAILURE;
  }

  // Load the plugin library.
  runtime->getPluginRegistry().loadLibrary(plugin_library_path.c_str());

  std::ifstream engine_file{engine_file_path, std::ios::binary};
  if (!engine_file) {
    std::cerr << "Failed to open the engine file." << std::endl;
    return EXIT_FAILURE;
  }

  engine_file.seekg(0, std::ios::end);
  std::size_t const engine_file_size{static_cast<std::size_t>(engine_file.tellg())};
  engine_file.seekg(0, std::ios::beg);

  std::unique_ptr<char[]> engine_data{new char[engine_file_size]};
  engine_file.read(engine_data.get(), engine_file_size);

  std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine{
    runtime->deserializeCudaEngine(engine_data.get(), engine_file_size)};
  if (engine == nullptr) {
    std::cerr << "Failed to deserialize the engine." << std::endl;
    return EXIT_FAILURE;
  }

  // Create the execution context.
  std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context{
    engine->createExecutionContext()};
  if (context == nullptr) {
    std::cerr << "Failed to create the execution context." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Engine loaded successfully." << std::endl;
  // std::this_thread::sleep_for(std::chrono::seconds(10));

  // Create the configuration
  const std::size_t cloud_capacity = 2000000;  // floats, not points
  const std::vector<int64_t> voxels_num{1, 128000, 256000};
  const std::vector<double> point_cloud_range{-122.4f, -122.4f, -3.f, 122.4f, 122.4f, 5.f};
  const std::vector<double> voxel_size{0.17f, 0.17f, 0.2f};
  const std::size_t num_proposals = 500;
  const float circle_nms_dist_threshold = 0.5f;
  const std::vector<double> & yaw_norm_thresholds{0.3f, 0.3f, 0.3f, 0.3f, 0.f};
  const float score_threshold = 0.1f;

  std::vector<double> dbound{1.0, 166.2, 1.4};

  std::vector<double> xbound{
    -122.4,
    122.4,
    0.68,
  };
  std::vector<double> ybound{
    -122.4,
    122.4,
    0.68,
  };
  std::vector<double> zbound{
    -10.0,
    10.0,
    20.0,
  };

  const std::size_t num_cameras = 6;
  const int raw_image_height = 1080;
  const int raw_image_width = 1440;

  const float img_aug_scale_x = 0.48f;
  const float img_aug_scale_y = 0.48f;
  const float img_aug_offset_y = -262.f;

  const int roi_height = 256;
  const int roi_width = 704;

  const int features_height = 32;  // Feature height
  const int features_width = 88;   // Feature width
  const int num_depth_features = 118;

  auto config = autoware::lidar_bevfusion::BEVFusionConfig(
    cloud_capacity, voxels_num, point_cloud_range, voxel_size, dbound, xbound, ybound, zbound,
    num_cameras, raw_image_height, raw_image_width, img_aug_scale_x, img_aug_scale_y,
    img_aug_offset_y, roi_height, roi_width, features_height, features_width, num_depth_features,
    num_proposals, circle_nms_dist_threshold, yaw_norm_thresholds, score_threshold);

  auto preprocessor = autoware::lidar_bevfusion::PreprocessCuda(config, stream, true);

  // Load images and attempt to do preprocessing

  int start_y = config.resized_height_ - config.roi_height_;
  int start_x =
    std::max(0, static_cast<int>(config.resized_width_) - static_cast<int>(config.roi_width_)) / 2;

  // IO tensor information and buffers.
  std::vector<nvinfer1::Dims> input_tensor_shapes{};
  std::vector<nvinfer1::Dims> output_tensor_shapes{};
  std::vector<std::size_t> input_tensor_sizes{};
  std::vector<std::size_t> output_tensor_sizes{};
  std::vector<char const *> input_tensor_names{};
  std::vector<char const *> output_tensor_names{};
  std::vector<void *> input_tensor_host_buffers{};
  std::vector<void *> input_tensor_device_buffers{};
  std::vector<void *> output_tensor_host_buffers{};
  std::vector<void *> output_tensor_device_buffers{};

  std::cout << "Loading and preprocessing images..." << std::endl;
  std::vector<std::uint8_t *> original_images_device(num_cameras);
  std::vector<std::uint8_t *> processed_images_device(num_cameras);
  std::vector<std::vector<std::uint8_t>> processed_images_host(num_cameras);
  std::uint8_t * processed_images_tensor_device;
  cudaMalloc(
    reinterpret_cast<void **>(&processed_images_tensor_device),
    num_cameras * config.roi_height_ * config.roi_width_ * 3 * sizeof(std::uint8_t));

  for (std::size_t camera_id = 0; camera_id < config.num_cameras_; camera_id++) {
    std::string image_file_name = "camera_" + std::to_string(camera_id) + "_original.png";
    cv::Mat image = cv::imread(image_file_name, cv::IMREAD_UNCHANGED);

    assert(static_cast<std::size_t>(image.rows) == config.raw_image_height_);
    assert(static_cast<std::size_t>(image.cols) == config.raw_image_width_);
    assert(
      static_cast<int>(image.total() * image.elemSize()) ==
      static_cast<int>(config.raw_image_height_ * config.raw_image_width_ * 3));

    cudaMalloc(
      reinterpret_cast<void **>(&original_images_device[camera_id]),
      config.raw_image_height_ * config.raw_image_width_ * 3);
    cudaMemcpy(
      original_images_device[camera_id], image.data,
      config.raw_image_height_ * config.raw_image_width_ * 3, cudaMemcpyHostToDevice);

    cudaMalloc(
      reinterpret_cast<void **>(&processed_images_device[camera_id]),
      config.roi_height_ * config.roi_width_ * 3);
    processed_images_host[camera_id].resize(config.roi_height_ * config.roi_width_ * 3);

    preprocessor.resize_and_extract_roi_launch(
      original_images_device[camera_id], processed_images_device[camera_id],
      config.raw_image_height_, config.raw_image_width_, config.resized_height_,
      config.resized_width_, config.roi_height_, config.roi_width_, start_y, start_x);
    cudaMemcpy(
      processed_images_host[camera_id].data(), processed_images_device[camera_id],
      config.roi_height_ * config.roi_width_ * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(
      processed_images_tensor_device + camera_id * config.roi_height_ * config.roi_width_ * 3,
      processed_images_device[camera_id],
      config.roi_height_ * config.roi_width_ * 3 * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    cv::Mat image_output(
      config.roi_height_, config.roi_width_, CV_8UC3, processed_images_host[camera_id].data());
    cv::imwrite("camera_" + std::to_string(camera_id) + "_processed.png", image_output);
  }

  std::cout << "Loading points and calibration..." << std::endl;
  std::vector<float> input_points_host = load_tensor<float>("points_test.txt");
  std::vector<float> cam2image_flattened_vector = load_tensor<float>("cam2image_test.txt");
  std::vector<float> camera2lidar_flattened_vector = load_tensor<float>("camera2lidar_test.txt");

  std::vector<Matrix4fRowM> cam2image_vector;
  std::vector<Matrix4fRowM> img_aug_vector;
  std::vector<Matrix4fRowM> img_aug_inverse_vector;
  std::vector<Matrix4fRowM> lidar2camera_vector;

  Matrix4fRowM img_aug_matrix = Matrix4fRowM::Identity();
  img_aug_matrix(0, 0) = config.img_aug_scale_x_;
  img_aug_matrix(1, 1) = config.img_aug_scale_y_;
  img_aug_matrix(1, 3) = config.img_aug_offset_y_;

  std::cout << "Preparing precomputed calibration tensors..." << std::endl;

  for (std::size_t camera_id = 0; camera_id < num_cameras; camera_id++) {
    Matrix4fRowM camera2lidar;
    Matrix4fRowM cam2image;

    for (std::size_t i = 0; i < 4; i++) {
      for (std::size_t j = 0; j < 4; j++) {
        camera2lidar(i, j) = camera2lidar_flattened_vector[camera_id * 16 + i * 4 + j];
        cam2image(i, j) = cam2image_flattened_vector[camera_id * 16 + i * 4 + j];
      }
    }

    lidar2camera_vector.push_back(camera2lidar.inverse());
    cam2image_vector.push_back(cam2image);
  }

  std::vector<sensor_msgs::msg::CameraInfo> camera_info_vector;

  for (std::size_t camera_id = 0; camera_id < num_cameras; camera_id++) {
    sensor_msgs::msg::CameraInfo camera_info;
    camera_info.p[0] = cam2image_vector[camera_id](0, 0);
    camera_info.p[5] = cam2image_vector[camera_id](1, 1);
    camera_info.p[2] = cam2image_vector[camera_id](0, 2);
    camera_info.p[6] = cam2image_vector[camera_id](1, 2);
    camera_info_vector.push_back(camera_info);
  }

  auto [lidar2images_flattened, geom_feats, kept, ranks, indices] =
    autoware::lidar_bevfusion::precompute_features(lidar2camera_vector, camera_info_vector, config);

  // use the new precomputed extrinsics+intrinsics
  std::vector<float> input_lidar2image_host(lidar2images_flattened.size());
  std::copy(
    lidar2images_flattened.data(), lidar2images_flattened.data() + lidar2images_flattened.size(),
    input_lidar2image_host.data());

  std::vector<std::int32_t> input_geom_feats_host(geom_feats.size());
  std::copy(geom_feats.data(), geom_feats.data() + geom_feats.size(), input_geom_feats_host.data());

  std::vector<std::uint8_t> input_kept_host(kept.size());
  std::copy(kept.data(), kept.data() + kept.size(), input_kept_host.data());

  std::vector<std::int64_t> input_ranks_host(ranks.size());
  std::copy(ranks.data(), ranks.data() + ranks.size(), input_ranks_host.data());

  std::vector<std::int64_t> input_indices_host(indices.size());
  std::copy(indices.data(), indices.data() + indices.size(), input_indices_host.data());

  // Attempt voxelization
  std::cout << "Voxelizing points..." << std::endl;
  float * points_device{};
  float * voxel_features_device{};
  std::int32_t * voxel_coords_device{};
  cudaMalloc(reinterpret_cast<void **>(&points_device), input_points_host.size() * sizeof(float));
  cudaMemcpy(
    points_device, input_points_host.data(), input_points_host.size() * sizeof(float),
    cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&voxel_features_device), 256000 * 5 * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&voxel_coords_device), 256000 * 4 * sizeof(std::int32_t));

  std::size_t num_voxels = preprocessor.generateVoxels(
    points_device, input_points_host.size() / 5, voxel_features_device, voxel_coords_device);

  std::vector<float> voxel_features_host(num_voxels * 5);
  std::vector<std::int32_t> voxel_coods_host(num_voxels * 4);

  cudaMemcpy(
    voxel_features_host.data(), voxel_features_device, num_voxels * 5 * sizeof(float),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(
    voxel_coods_host.data(), voxel_coords_device, num_voxels * 4 * sizeof(std::int32_t),
    cudaMemcpyDeviceToHost);

  float min_voxel_feature_x = 1e6;
  float min_voxel_feature_y = 1e6;
  float min_voxel_feature_z = 1e6;
  float max_voxel_feature_x = -1e6;
  float max_voxel_feature_y = -1e6;
  float max_voxel_feature_z = -1e6;

  std::int32_t min_voxel_coord_b = 1e6;
  std::int32_t min_voxel_coord_x = 1e6;
  std::int32_t min_voxel_coord_y = 1e6;
  std::int32_t min_voxel_coord_z = 1e6;
  std::int32_t max_voxel_coord_b = -1e6;
  std::int32_t max_voxel_coord_x = -1e6;
  std::int32_t max_voxel_coord_y = -1e6;
  std::int32_t max_voxel_coord_z = -1e6;

  for (std::size_t voxel_id = 0; voxel_id < num_voxels; voxel_id++) {
    min_voxel_feature_x = std::min(min_voxel_feature_x, voxel_features_host[voxel_id * 5 + 0]);
    min_voxel_feature_y = std::min(min_voxel_feature_y, voxel_features_host[voxel_id * 5 + 1]);
    min_voxel_feature_z = std::min(min_voxel_feature_z, voxel_features_host[voxel_id * 5 + 2]);
    max_voxel_feature_x = std::max(max_voxel_feature_x, voxel_features_host[voxel_id * 5 + 0]);
    max_voxel_feature_y = std::max(max_voxel_feature_y, voxel_features_host[voxel_id * 5 + 1]);
    max_voxel_feature_z = std::max(max_voxel_feature_z, voxel_features_host[voxel_id * 5 + 2]);

    min_voxel_coord_b = std::min(min_voxel_coord_b, voxel_coods_host[voxel_id * 4 + 0]);
    min_voxel_coord_x = std::min(min_voxel_coord_x, voxel_coods_host[voxel_id * 4 + 1]);
    min_voxel_coord_y = std::min(min_voxel_coord_y, voxel_coods_host[voxel_id * 4 + 2]);
    min_voxel_coord_z = std::min(min_voxel_coord_z, voxel_coods_host[voxel_id * 4 + 3]);
    max_voxel_coord_b = std::max(max_voxel_coord_b, voxel_coods_host[voxel_id * 4 + 0]);
    max_voxel_coord_x = std::max(max_voxel_coord_x, voxel_coods_host[voxel_id * 4 + 1]);
    max_voxel_coord_y = std::max(max_voxel_coord_y, voxel_coods_host[voxel_id * 4 + 2]);
    max_voxel_coord_z = std::max(max_voxel_coord_z, voxel_coods_host[voxel_id * 4 + 3]);
  }

  std::cout << "num_voxels: " << num_voxels << std::endl;

  std::cout << "Preparing input tensors..." << std::endl;

  nvinfer1::Dims input_voxels_shape;
  input_voxels_shape.nbDims = 2;
  input_voxels_shape.d[0] = num_voxels;
  input_voxels_shape.d[1] = 5;

  nvinfer1::Dims input_coors_shape;
  input_coors_shape.nbDims = 2;
  input_coors_shape.d[0] = num_voxels;
  input_coors_shape.d[1] = 4;

  nvinfer1::Dims input_imgs_shape;
  input_imgs_shape.nbDims = 4;
  input_imgs_shape.d[0] = config.num_cameras_;
  input_imgs_shape.d[1] = 3;
  input_imgs_shape.d[2] = config.roi_height_;
  input_imgs_shape.d[3] = config.roi_width_;

  nvinfer1::Dims input_lidar2image_shape;
  input_lidar2image_shape.nbDims = 3;
  input_lidar2image_shape.d[0] = config.num_cameras_;
  input_lidar2image_shape.d[1] = 4;
  input_lidar2image_shape.d[2] = 4;
  assert(input_lidar2image_host.size() == config.num_cameras_ * 4 * 4);

  nvinfer1::Dims input_geom_feats_shape;
  input_geom_feats_shape.nbDims = 2;
  input_geom_feats_shape.d[0] = input_geom_feats_host.size() / 4;
  input_geom_feats_shape.d[1] = 4;

  nvinfer1::Dims input_kept_shape;
  input_kept_shape.nbDims = 1;
  input_kept_shape.d[0] = input_kept_host.size();

  nvinfer1::Dims input_ranks_shape;
  input_ranks_shape.nbDims = 1;
  input_ranks_shape.d[0] = input_ranks_host.size();

  nvinfer1::Dims input_indices_shape;
  input_indices_shape.nbDims = 1;
  input_indices_shape.d[0] = input_indices_host.size();

  assert(input_indices_host.size() == input_ranks_host.size());
  assert(input_indices_host.size() == input_geom_feats_host.size() / 4);

  std::vector<float> output_bbox_pred_host(10 * num_proposals);
  std::vector<float> output_score_host(num_proposals);
  std::vector<std::int64_t> output_label_pred_host(num_proposals);

  nvinfer1::Dims output_bbox_pred_shape;
  output_bbox_pred_shape.nbDims = 2;
  output_bbox_pred_shape.d[0] = 10;
  output_bbox_pred_shape.d[1] = config.num_proposals_;

  nvinfer1::Dims output_score_shape;
  output_score_shape.nbDims = 1;
  output_score_shape.d[0] = config.num_proposals_;

  nvinfer1::Dims output_label_pred_shape;
  output_label_pred_shape.nbDims = 1;
  output_label_pred_shape.d[0] = config.num_proposals_;

  std::unordered_map<std::string, void *> input_tensor_host_buffers_map;
  std::unordered_map<std::string, void *> input_tensor_device_buffers_map;
  std::unordered_map<std::string, nvinfer1::Dims> input_shapes_map;
  std::unordered_map<std::string, std::size_t> input_sizes_map;

  std::unordered_map<std::string, void *> output_tensor_host_buffers_map;
  std::unordered_map<std::string, void *> output_tensor_device_buffers_map;
  std::unordered_map<std::string, nvinfer1::Dims> output_shapes_map;
  std::unordered_map<std::string, std::size_t> output_sizes_map;

  input_sizes_map["lidar2image"] = input_lidar2image_host.size() * sizeof(float);
  input_sizes_map["geom_feats"] = input_geom_feats_host.size() * sizeof(std::int32_t);
  input_sizes_map["kept"] = input_kept_host.size() * sizeof(std::uint8_t);
  input_sizes_map["ranks"] = input_ranks_host.size() * sizeof(std::int64_t);
  input_sizes_map["indices"] = input_indices_host.size() * sizeof(std::int64_t);

  input_tensor_host_buffers_map["lidar2image"];
  cudaMallocHost(&input_tensor_host_buffers_map["lidar2image"], input_sizes_map["lidar2image"]);
  memcpy(
    input_tensor_host_buffers_map["lidar2image"], input_lidar2image_host.data(),
    input_sizes_map["lidar2image"]);

  input_tensor_host_buffers_map["geom_feats"];
  cudaMallocHost(&input_tensor_host_buffers_map["geom_feats"], input_sizes_map["geom_feats"]);
  memcpy(
    input_tensor_host_buffers_map["geom_feats"], input_geom_feats_host.data(),
    input_sizes_map["geom_feats"]);

  input_tensor_host_buffers_map["kept"];
  cudaMallocHost(&input_tensor_host_buffers_map["kept"], input_sizes_map["kept"]);
  memcpy(input_tensor_host_buffers_map["kept"], input_kept_host.data(), input_sizes_map["kept"]);

  input_tensor_host_buffers_map["ranks"];
  cudaMallocHost(&input_tensor_host_buffers_map["ranks"], input_sizes_map["ranks"]);
  memcpy(input_tensor_host_buffers_map["ranks"], input_ranks_host.data(), input_sizes_map["ranks"]);

  input_tensor_host_buffers_map["indices"];
  cudaMallocHost(&input_tensor_host_buffers_map["indices"], input_sizes_map["indices"]);
  memcpy(
    input_tensor_host_buffers_map["indices"], input_indices_host.data(),
    input_sizes_map["indices"]);

  input_shapes_map["voxels"] = input_voxels_shape;
  input_shapes_map["coors"] = input_coors_shape;
  input_shapes_map["imgs"] = input_imgs_shape;
  input_shapes_map["lidar2image"] = input_lidar2image_shape;
  input_shapes_map["geom_feats"] = input_geom_feats_shape;
  input_shapes_map["kept"] = input_kept_shape;
  input_shapes_map["ranks"] = input_ranks_shape;
  input_shapes_map["indices"] = input_indices_shape;

  CHECK_CUDA_ERROR(
    cudaMalloc(&input_tensor_device_buffers_map["voxels"], input_sizes_map["voxels"]));
  CHECK_CUDA_ERROR(cudaMalloc(&input_tensor_device_buffers_map["coors"], input_sizes_map["coors"]));
  CHECK_CUDA_ERROR(cudaMalloc(&input_tensor_device_buffers_map["imgs"], input_sizes_map["imgs"]));
  CHECK_CUDA_ERROR(
    cudaMalloc(&input_tensor_device_buffers_map["lidar2image"], input_sizes_map["lidar2image"]));
  CHECK_CUDA_ERROR(
    cudaMalloc(&input_tensor_device_buffers_map["geom_feats"], input_sizes_map["geom_feats"]));
  CHECK_CUDA_ERROR(cudaMalloc(&input_tensor_device_buffers_map["kept"], input_sizes_map["kept"]));
  CHECK_CUDA_ERROR(cudaMalloc(&input_tensor_device_buffers_map["ranks"], input_sizes_map["ranks"]));
  CHECK_CUDA_ERROR(
    cudaMalloc(&input_tensor_device_buffers_map["indices"], input_sizes_map["indices"]));

  output_sizes_map["bbox_pred"] = output_bbox_pred_host.size() * sizeof(float);
  output_sizes_map["score"] = output_score_host.size() * sizeof(float);
  output_sizes_map["label_pred"] = output_label_pred_host.size() * sizeof(std::int64_t);

  cudaMallocHost(&output_tensor_host_buffers_map["bbox_pred"], output_sizes_map["bbox_pred"]);
  memcpy(
    output_tensor_host_buffers_map["bbox_pred"], output_bbox_pred_host.data(),
    output_sizes_map["bbox_pred"]);

  cudaMallocHost(&output_tensor_host_buffers_map["score"], output_sizes_map["score"]);
  memcpy(
    output_tensor_host_buffers_map["score"], output_score_host.data(), output_sizes_map["score"]);

  cudaMallocHost(&output_tensor_host_buffers_map["label_pred"], output_sizes_map["label_pred"]);
  memcpy(
    output_tensor_host_buffers_map["label_pred"], output_label_pred_host.data(),
    output_sizes_map["label_pred"]);

  output_shapes_map["bbox_pred"] = output_bbox_pred_shape;
  output_shapes_map["score"] = output_score_shape;
  output_shapes_map["label_pred"] = output_label_pred_shape;

  CHECK_CUDA_ERROR(
    cudaMalloc(&output_tensor_device_buffers_map["bbox_pred"], output_sizes_map["bbox_pred"]));
  CHECK_CUDA_ERROR(
    cudaMalloc(&output_tensor_device_buffers_map["score"], output_sizes_map["score"]));
  CHECK_CUDA_ERROR(
    cudaMalloc(&output_tensor_device_buffers_map["label_pred"], output_sizes_map["label_pred"]));

  for (auto & [key, value] : input_sizes_map) {
    cudaError_t err = cudaMemcpy(
      input_tensor_device_buffers_map[key], input_tensor_host_buffers_map[key], value,
      cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  for (auto & [key, value] : output_sizes_map) {
    cudaError_t err = cudaMemcpy(
      output_tensor_device_buffers_map[key], output_tensor_host_buffers_map[key], value,
      cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // These were aready in gpu
  input_tensor_device_buffers_map["voxels"] = voxel_features_device;
  input_tensor_device_buffers_map["coors"] = voxel_coords_device;
  input_tensor_device_buffers_map["imgs"] = processed_images_tensor_device;

  // Check the number of IO tensors.
  std::int32_t const num_io_tensors{engine->getNbIOTensors()};
  std::cout << "Number of IO Tensors: " << num_io_tensors << std::endl;
  for (std::int32_t i{0}; i < num_io_tensors; ++i) {
    char const * const tensor_name{engine->getIOTensorName(i)};
    std::cout << "Tensor name: " << tensor_name << std::endl;

    nvinfer1::TensorFormat const format{engine->getTensorFormat(tensor_name)};
    if (format != expected_format) {
      std::cerr << "Invalid tensor format." << std::endl;
      return EXIT_FAILURE;
    }
    // Because the input and output shapes are static,
    // there is no need to set the IO tensor shapes.
    nvinfer1::Dims const shape{engine->getTensorShape(tensor_name)};
    // Print out dims.
    std::size_t tensor_size{1U};
    std::cout << "Tensor Dims: ";
    for (std::int32_t j{0}; j < shape.nbDims; ++j) {
      tensor_size *= shape.d[j];
      std::cout << shape.d[j] << " ";
    }
    std::cout << std::endl;
  }

  // Inputs
  context->setTensorAddress("voxels", input_tensor_device_buffers_map["voxels"]);
  context->setTensorAddress("coors", input_tensor_device_buffers_map["coors"]);
  context->setTensorAddress("imgs", input_tensor_device_buffers_map["imgs"]);
  context->setTensorAddress("lidar2image", input_tensor_device_buffers_map["lidar2image"]);
  context->setTensorAddress("geom_feats", input_tensor_device_buffers_map["geom_feats"]);
  context->setTensorAddress("kept", input_tensor_device_buffers_map["kept"]);
  context->setTensorAddress("ranks", input_tensor_device_buffers_map["ranks"]);
  context->setTensorAddress("indices", input_tensor_device_buffers_map["indices"]);

  context->setInputShape("voxels", input_shapes_map["voxels"]);
  context->setInputShape("coors", input_shapes_map["coors"]);
  context->setInputShape("imgs", input_shapes_map["imgs"]);
  context->setInputShape("lidar2image", input_shapes_map["lidar2image"]);
  context->setInputShape("geom_feats", input_shapes_map["geom_feats"]);
  context->setInputShape("kept", input_shapes_map["kept"]);
  context->setInputShape("ranks", input_shapes_map["ranks"]);
  context->setInputShape("indices", input_shapes_map["indices"]);

  // Outputs
  context->setTensorAddress("bbox_pred", output_tensor_device_buffers_map["bbox_pred"]);
  context->setTensorAddress("score", output_tensor_device_buffers_map["score"]);
  context->setTensorAddress("label_pred", output_tensor_device_buffers_map["label_pred"]);

  std::cout << "Running inference..." << std::endl;

  // Run inference a couple of times.
  std::size_t const num_iterations{1};
  for (std::size_t i{0U}; i < num_iterations; ++i) {
    bool const status{context->enqueueV3(stream)};
    if (!status) {
      std::cerr << "Failed to run inference." << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Synchronize.
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  std::cout << "Copying data to host..." << std::endl;

  for (auto & [key, value] : output_sizes_map) {
    cudaError_t err = cudaMemcpy(
      output_tensor_host_buffers_map[key], output_tensor_device_buffers_map[key], value,
      cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  [[maybe_unused]] auto output_bbox_pred_real_shape = context->getTensorShape("bbox_pred");
  [[maybe_unused]] auto output_score_real_shape = context->getTensorShape("score");
  [[maybe_unused]] auto output_label_pred_real_shape = context->getTensorShape("label_pred");

  memcpy(
    output_bbox_pred_host.data(), output_tensor_host_buffers_map["bbox_pred"],
    output_sizes_map["bbox_pred"]);
  memcpy(
    output_score_host.data(), output_tensor_host_buffers_map["score"], output_sizes_map["score"]);
  memcpy(
    output_label_pred_host.data(), output_tensor_host_buffers_map["label_pred"],
    output_sizes_map["label_pred"]);

  for (std::size_t i = 0; i < 20; i++) {
    std::cout << "i: " << i << " output_score_host20: " << output_score_host[i] << std::endl;
  }

  // Release resources.
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
  for (std::size_t i{0U}; i < input_tensor_host_buffers.size(); ++i) {
    CHECK_CUDA_ERROR(cudaFreeHost(input_tensor_host_buffers.at(i)));
  }
  for (std::size_t i{0U}; i < input_tensor_device_buffers.size(); ++i) {
    CHECK_CUDA_ERROR(cudaFree(input_tensor_device_buffers.at(i)));
  }
  for (std::size_t i{0U}; i < output_tensor_host_buffers.size(); ++i) {
    CHECK_CUDA_ERROR(cudaFreeHost(output_tensor_host_buffers.at(i)));
  }
  for (std::size_t i{0U}; i < output_tensor_device_buffers.size(); ++i) {
    CHECK_CUDA_ERROR(cudaFree(output_tensor_device_buffers.at(i)));
  }

  return 0;
}
