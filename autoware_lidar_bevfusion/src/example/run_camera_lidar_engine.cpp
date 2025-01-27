// Copyright 2025 TIER IV, Inc.
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
#include <spconvlib/spconv/csrc/sparse/all/ops3d/Point2Voxel.h>

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

class MyProfiler : public nvinfer1::IProfiler
{
public:
  void reportLayerTime(const char * layerName, float ms) noexcept override
  {
    if (std::string(layerName).find("ImplicitGemm") != std::string::npos) {
      //&& std::string(layerName).find("GetIndicePairs") == std::string::npos)
      std::cout << "Layer: " << layerName << " took " << ms << " ms" << std::endl;
      total_ms += ms;
    }
  }

  float total_ms{0.f};
};

int main(int argc, char ** argv)
{
  using autoware::lidar_bevfusion::Matrix4fRowM;
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <plugin_library_path> <engine_file_path> <camera_lidar_mode>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string const plugin_library_path{argv[1]};
  std::string const engine_file_path{argv[2]};
  bool const camera_lidar_mode = std::stoi(argv[3]);

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

  MyProfiler profiler;
  // context->setProfiler(&profiler);

  std::cout << "Engine loaded successfully." << std::endl;

  // Create the configuration
  // const int max_points_per_voxel = 10;
  const int64_t cloud_capacity = 2000000;  // floats, not points
  const std::vector<int64_t> voxels_num{1, 128000, 256000};
  const std::vector<float> point_cloud_range{-122.4f, -122.4f, -3.f, 122.4f, 122.4f, 5.f};
  const std::vector<float> voxel_size{0.17f, 0.17f, 0.2f};
  const int64_t num_proposals = 500;
  const float circle_nms_dist_threshold = 0.5f;
  const std::vector<double> & yaw_norm_thresholds{0.3f, 0.3f, 0.3f, 0.3f, 0.f};
  const float score_threshold = 0.1f;

  std::cout << "num_proposals: " << num_proposals << std::endl;

  std::vector<float> dbound{1.0, 166.2, 1.4};

  std::vector<float> xbound{
    -122.4,
    122.4,
    0.68,
  };
  std::vector<float> ybound{
    -122.4,
    122.4,
    0.68,
  };
  std::vector<float> zbound{
    -10.0,
    10.0,
    20.0,
  };

  const std::size_t num_cameras = 6;
  const int raw_image_height = 1080;
  const int raw_image_width = 1440;

  // TODO(knzo25): these values will be computed automatically in the node, and preloaded here, so
  // these should not be used
  const float img_aug_scale_x = 0.489f;
  const float img_aug_scale_y = 0.489f;
  [[maybe_unused]] const float img_aug_offset_y = -262.f;

  const int roi_height = 384;  // 256;
  const int roi_width = 704;

  const int features_height = 48;  // 32;  // Feature height
  const int features_width = 88;   // Feature width
  const int num_depth_features = 118;

  const std::vector<float> camera_mask{1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

  auto config = autoware::lidar_bevfusion::BEVFusionConfig(
    camera_lidar_mode, "", 8, cloud_capacity, 10, voxels_num, point_cloud_range, voxel_size, dbound,
    xbound, ybound, zbound, num_cameras, raw_image_height, raw_image_width, img_aug_scale_x,
    img_aug_scale_y, roi_height, roi_width, features_height, features_width, num_depth_features,
    num_proposals, circle_nms_dist_threshold, yaw_norm_thresholds, score_threshold);

  std::cout << "config.config.num_proposals_: " << config.num_proposals_ << std::endl;

  auto preprocessor = autoware::lidar_bevfusion::PreprocessCuda(config, stream, true);

  // Load images and attempt to do preprocessing

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

  std::cout << "Loading points and calibration..." << std::endl;
  std::vector<float> input_points_host = load_tensor<float>("points.txt");
  std::vector<float> input_voxels_host = load_tensor<float>("feats.txt");
  std::vector<std::int32_t> input_coors_host = load_tensor<std::int32_t>("coors.txt");
  std::vector<float> cam2image_flattened_vector = load_tensor<float>("cam2image.txt");
  std::vector<float> camera2lidar_flattened_vector = load_tensor<float>("camera2lidar.txt");
  std::vector<float> img_aug_flattened_vector = load_tensor<float>("img_aug.txt");

  std::vector<Matrix4fRowM> cam2image_vector;
  std::vector<Matrix4fRowM> img_aug_vector;
  std::vector<Matrix4fRowM> lidar2camera_vector;

  std::cout << "num_points: " << input_points_host.size() / 5 << std::endl;

  /* Matrix4fRowM img_aug_matrix = Matrix4fRowM::Identity();
  img_aug_matrix(0, 0) = config.img_aug_scale_x_;
  img_aug_matrix(1, 1) = config.img_aug_scale_y_;
  img_aug_matrix(1, 3) = config.img_aug_offset_y_; */

  std::cout << "Preparing precomputed calibration tensors..." << std::endl;

  for (std::size_t camera_id = 0; camera_id < num_cameras; camera_id++) {
    Matrix4fRowM camera2lidar;
    Matrix4fRowM cam2image;
    Matrix4fRowM img_aug;

    for (std::size_t i = 0; i < 4; i++) {
      for (std::size_t j = 0; j < 4; j++) {
        camera2lidar(i, j) = camera2lidar_flattened_vector[camera_id * 16 + i * 4 + j];
        cam2image(i, j) = cam2image_flattened_vector[camera_id * 16 + i * 4 + j];
        img_aug(i, j) = img_aug_flattened_vector[camera_id * 16 + i * 4 + j];
      }
    }

    lidar2camera_vector.push_back(camera2lidar.inverse());
    cam2image_vector.push_back(cam2image);
    img_aug_vector.push_back(img_aug);
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

  std::cout << "Loading and preprocessing images..." << std::endl;
  std::vector<std::uint8_t *> original_images_device(num_cameras);
  std::vector<std::uint8_t *> processed_images_device(num_cameras);
  std::vector<std::vector<std::uint8_t>> processed_images_host(num_cameras);
  std::uint8_t * processed_images_tensor_device;
  cudaMalloc(
    reinterpret_cast<void **>(&processed_images_tensor_device),
    num_cameras * config.roi_height_ * config.roi_width_ * 3 * sizeof(std::uint8_t));

  for (std::int64_t camera_id = 0; camera_id < config.num_cameras_; camera_id++) {
    std::string image_file_name = "camera_" + std::to_string(camera_id) + "_original.png";
    cv::Mat image = cv::imread(image_file_name, cv::IMREAD_UNCHANGED);
    std::cout << "Image size: " << image.size() << std::endl;

    assert(static_cast<std::int64_t>(image.rows) == config.raw_image_height_);
    assert(static_cast<std::int64_t>(image.cols) == config.raw_image_width_);
    assert(
      static_cast<int>(image.total() * image.elemSize()) ==
      static_cast<int>(config.raw_image_height_ * config.raw_image_width_ * 3));

    // int start_y = config.resized_height_ - config.roi_height_;
    int start_x =
      std::max(0, static_cast<int>(config.resized_width_) - static_cast<int>(config.roi_width_)) /
      2;

    int start_y = -img_aug_vector[camera_id](1, 3);

    std::cout << "Resizing and extracting ROI..." << std::endl;
    std::cout << "start_y: " << start_y << std::endl;
    std::cout << "start_x: " << start_x << std::endl;
    std::cout << img_aug_vector[camera_id] << std::endl;

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
      config.resized_width_, config.roi_height_, config.roi_width_, start_y, start_x, stream);
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

  auto [lidar2images_flattened, geom_feats, kept, ranks, indices] =
    autoware::lidar_bevfusion::precompute_features(
      lidar2camera_vector, img_aug_vector, camera_info_vector, config);

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
  std::int32_t * num_points_per_voxel_device{};
  cudaMalloc(reinterpret_cast<void **>(&points_device), input_points_host.size() * sizeof(float));
  cudaMemcpy(
    points_device, input_points_host.data(), input_points_host.size() * sizeof(float),
    cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&voxel_features_device), 256000 * 10 * 5 * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&voxel_coords_device), 256000 * 3 * sizeof(std::int32_t));
  cudaMalloc(
    reinterpret_cast<void **>(&num_points_per_voxel_device), 256000 * sizeof(std::int32_t));

  /* std::size_t num_voxels = input_voxels_host.size() / 5;
  assert(num_voxels == input_coors_host.size() / 4);
  cudaMemcpy(
    voxel_features_device, input_voxels_host.data(), num_voxels * 5 * sizeof(float),
    cudaMemcpyHostToDevice);
  cudaMemcpy(
    voxel_coords_device, input_coors_host.data(), num_voxels * 4 * sizeof(std::int32_t),
    cudaMemcpyHostToDevice); */

  std::size_t num_voxels;
  const int voxelization_iterations = 50;
  float avg_voxelization_time_ms = 0.0f;

  for (int iteration = 0; iteration < voxelization_iterations; iteration++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    num_voxels = preprocessor.generateVoxels(
      points_device, input_points_host.size() / 5, voxel_features_device, voxel_coords_device,
      num_points_per_voxel_device);

    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto voxelization_time_ms =
      0.001f * std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    avg_voxelization_time_ms += iteration > 0 ? voxelization_time_ms : 0.f;

    std::cout << "Voxelization took " << voxelization_time_ms << " ms" << std::endl;
  }

  std::cout << "Average voxelization time: "
            << avg_voxelization_time_ms / (voxelization_iterations - 1) << " ms" << std::endl;

  std::cout << "num_voxels: " << num_voxels << std::endl;

  std::cout << "Preparing input tensors..." << std::endl;

  nvinfer1::Dims input_voxels_shape;
  input_voxels_shape.nbDims = 3;
  input_voxels_shape.d[0] = num_voxels;
  input_voxels_shape.d[1] = 10;
  input_voxels_shape.d[2] = 5;

  nvinfer1::Dims input_coors_shape;
  input_coors_shape.nbDims = 2;
  input_coors_shape.d[0] = num_voxels;
  input_coors_shape.d[1] = 3;

  nvinfer1::Dims input_num_points_per_voxel_shape;
  input_num_points_per_voxel_shape.nbDims = 1;
  input_num_points_per_voxel_shape.d[0] = num_voxels;

  nvinfer1::Dims input_points_shape;
  input_points_shape.nbDims = 2;
  input_points_shape.d[0] = input_points_host.size() / 5;
  input_points_shape.d[1] = 5;

  nvinfer1::Dims input_cameras_shape;
  input_cameras_shape.nbDims = 1;
  input_cameras_shape.d[0] = num_cameras;

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
  assert(static_cast<std::int64_t>(input_lidar2image_host.size()) == config.num_cameras_ * 4 * 4);

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

  input_sizes_map["camera_mask"] = camera_mask.size() * sizeof(float);
  input_sizes_map["lidar2image"] = input_lidar2image_host.size() * sizeof(float);
  input_sizes_map["geom_feats"] = input_geom_feats_host.size() * sizeof(std::int32_t);
  input_sizes_map["kept"] = input_kept_host.size() * sizeof(std::uint8_t);
  input_sizes_map["ranks"] = input_ranks_host.size() * sizeof(std::int64_t);
  input_sizes_map["indices"] = input_indices_host.size() * sizeof(std::int64_t);

  input_tensor_host_buffers_map["camera_mask"];
  cudaMallocHost(&input_tensor_host_buffers_map["camera_mask"], input_sizes_map["camera_mask"]);
  memcpy(
    input_tensor_host_buffers_map["camera_mask"], camera_mask.data(),
    input_sizes_map["camera_mask"]);

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
  input_shapes_map["num_points_per_voxel"] = input_num_points_per_voxel_shape;
  input_shapes_map["points"] = input_points_shape;
  input_shapes_map["camera_mask"] = input_cameras_shape;
  input_shapes_map["imgs"] = input_imgs_shape;
  input_shapes_map["lidar2image"] = input_lidar2image_shape;
  input_shapes_map["geom_feats"] = input_geom_feats_shape;
  input_shapes_map["kept"] = input_kept_shape;
  input_shapes_map["ranks"] = input_ranks_shape;
  input_shapes_map["indices"] = input_indices_shape;

  CHECK_CUDA_ERROR(
    cudaMalloc(&input_tensor_device_buffers_map["voxels"], input_sizes_map["voxels"]));
  CHECK_CUDA_ERROR(cudaMalloc(&input_tensor_device_buffers_map["coors"], input_sizes_map["coors"]));
  CHECK_CUDA_ERROR(cudaMalloc(
    &input_tensor_device_buffers_map["num_points_per_voxel"],
    input_sizes_map["num_points_per_voxel"]));
  CHECK_CUDA_ERROR(
    cudaMalloc(&input_tensor_device_buffers_map["points"], input_sizes_map["points"]));
  CHECK_CUDA_ERROR(
    cudaMalloc(&input_tensor_device_buffers_map["camera_mask"], input_sizes_map["camera_mask"]));
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
  input_tensor_device_buffers_map["num_points_per_voxel"] = num_points_per_voxel_device;
  input_tensor_device_buffers_map["points"] = points_device;
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
  context->setTensorAddress(
    "num_points_per_voxel", input_tensor_device_buffers_map["num_points_per_voxel"]);

  context->setInputShape("voxels", input_shapes_map["voxels"]);
  context->setInputShape("coors", input_shapes_map["coors"]);
  context->setInputShape("num_points_per_voxel", input_shapes_map["num_points_per_voxel"]);

  if (camera_lidar_mode) {
    context->setTensorAddress("points", input_tensor_device_buffers_map["points"]);
    context->setTensorAddress("camera_mask", input_tensor_device_buffers_map["camera_mask"]);
    context->setTensorAddress("imgs", input_tensor_device_buffers_map["imgs"]);
    context->setTensorAddress("lidar2image", input_tensor_device_buffers_map["lidar2image"]);
    context->setTensorAddress("geom_feats", input_tensor_device_buffers_map["geom_feats"]);
    context->setTensorAddress("kept", input_tensor_device_buffers_map["kept"]);
    context->setTensorAddress("ranks", input_tensor_device_buffers_map["ranks"]);
    context->setTensorAddress("indices", input_tensor_device_buffers_map["indices"]);

    context->setInputShape("points", input_shapes_map["points"]);
    context->setInputShape("camera_mask", input_shapes_map["camera_mask"]);
    context->setInputShape("imgs", input_shapes_map["imgs"]);
    context->setInputShape("lidar2image", input_shapes_map["lidar2image"]);
    context->setInputShape("geom_feats", input_shapes_map["geom_feats"]);
    context->setInputShape("kept", input_shapes_map["kept"]);
    context->setInputShape("ranks", input_shapes_map["ranks"]);
    context->setInputShape("indices", input_shapes_map["indices"]);
  }
  // Outputs
  context->setTensorAddress("bbox_pred", output_tensor_device_buffers_map["bbox_pred"]);
  context->setTensorAddress("score", output_tensor_device_buffers_map["score"]);
  context->setTensorAddress("label_pred", output_tensor_device_buffers_map["label_pred"]);

  std::cout << "Running inference..." << std::endl;

  const int inference_iterations = 50;
  float avg_inference_time_ms = 0.0f;

  // Run inference a couple of times.
  for (std::size_t i{0U}; i < inference_iterations; ++i) {
    profiler.total_ms = 0.f;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool const status{context->enqueueV3(stream)};
    if (!status) {
      std::cerr << "Failed to run inference." << std::endl;
      return EXIT_FAILURE;
    }

    // Synchronize.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    avg_inference_time_ms += i > 0 ? dt_ms : 0.f;
    std::cout << "Inference time: " << dt_ms << " ms" << std::endl;
    std::cout << "Sparse convolution time: " << profiler.total_ms << " ms" << std::endl;
  }

  std::cout << "Average inference time: " << avg_inference_time_ms / (inference_iterations - 1)
            << " ms" << std::endl;

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

  std::cout << "Output bbox pred shape: " << output_bbox_pred_real_shape.nbDims << std::endl;

  int real_detections = output_score_real_shape.d[0];

  for (int i = 0; i < real_detections; i++) {
    std::cout << "Detection i: " << i << ": ";
    for (int j = 0; j < 10; j++) {
      std::cout << output_bbox_pred_host[j * real_detections + i] << " ";
    }
    std::cout << std::endl;
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
