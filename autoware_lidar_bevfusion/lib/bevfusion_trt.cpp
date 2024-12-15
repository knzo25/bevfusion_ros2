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

#include "autoware/lidar_bevfusion/bevfusion_trt.hpp"

#include "autoware/lidar_bevfusion/bevfusion_config.hpp"
#include "autoware/lidar_bevfusion/preprocess/precomputed_features.hpp"
#include "autoware/lidar_bevfusion/preprocess/preprocess_kernel.hpp"

#include <autoware/universe_utils/math/constants.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace autoware::lidar_bevfusion
{

BEVFusionTRT::BEVFusionTRT(
  const NetworkParam & network_param, const DensificationParam & densification_param,
  const BEVFusionConfig & config)
: config_(config)
{
  network_trt_ptr_ = std::make_unique<NetworkTRT>(config_);

  network_trt_ptr_->init(
    network_param.onnx_path(), network_param.engine_path(), network_param.trt_precision());

  vg_ptr_ = std::make_unique<VoxelGenerator>(densification_param, config_, stream_);

  stop_watch_ptr_ =
    std::make_unique<autoware::universe_utils::StopWatch<std::chrono::milliseconds>>();
  stop_watch_ptr_->tic("processing/inner");

  initPtr();

  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

BEVFusionTRT::~BEVFusionTRT()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
}

void BEVFusionTRT::initPtr()
{
  // point cloud to voxels
  voxel_features_size_ = 5 * config_.max_voxels_;
  voxel_coords_size_ = 4 * config_.max_voxels_;

  // output of TRT -- input of post-process
  bbox_pred_size_ = config_.num_proposals_ * config_.num_box_values_;
  label_pred_output_d_ = cuda::make_unique<std::int64_t[]>(config_.num_proposals_);
  bbox_pred_output_d_ = cuda::make_unique<float[]>(bbox_pred_size_);
  score_output_d_ = cuda::make_unique<float[]>(config_.num_proposals_);

  // lidar branch
  voxel_features_d_ = cuda::make_unique<float[]>(voxel_features_size_);
  voxel_coords_d_ = cuda::make_unique<std::int32_t[]>(voxel_coords_size_);
  num_points_per_voxel_d_ = cuda::make_unique<std::int32_t[]>(config_.max_voxels_);
  points_d_ = cuda::make_unique<float[]>(config_.cloud_capacity_ * config_.num_point_feature_size_);

  // pre computed tensors
  lidar2image_d_ = cuda::make_unique<float[]>(config_.num_cameras_ * 4 * 4);
  std::size_t num_geom_feats = config_.num_cameras_ * config_.features_height_ *
                               config_.features_width_ * config_.num_depth_features_;
  geom_feats_d_ = cuda::make_unique<std::int32_t[]>(4 * num_geom_feats);
  kept_d_ = cuda::make_unique<std::uint8_t[]>(num_geom_feats);
  ranks_d_ = cuda::make_unique<std::int64_t[]>(num_geom_feats);
  indices_d_ = cuda::make_unique<std::int64_t[]>(num_geom_feats);

  // image branch
  roi_tensor_d_ = cuda::make_unique<std::uint8_t[]>(
    config_.num_cameras_ * config_.roi_height_ * config_.roi_width_ * 3);
  for (std::size_t camera_id = 0; camera_id < config_.num_cameras_; camera_id++) {
    image_buffers_d_.emplace_back(
      cuda::make_unique<std::uint8_t[]>(config_.raw_image_height_ * config_.raw_image_width_ * 3));
  }

  pre_ptr_ = std::make_unique<PreprocessCuda>(config_, stream_, true);
  post_ptr_ = std::make_unique<PostprocessCuda>(config_, stream_);
}

bool BEVFusionTRT::detect(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pc_msg,
  const std::vector<sensor_msgs::msg::Image::ConstSharedPtr> & image_msgs,
  const tf2_ros::Buffer & tf_buffer, std::vector<Box3D> & det_boxes3d,
  std::unordered_map<std::string, double> & proc_timing)
{
  stop_watch_ptr_->toc("processing/inner", true);
  if (!preprocess(pc_msg, image_msgs, tf_buffer)) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_bevfusion"), "Fail to preprocess and skip to detect.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/preprocess_ms", stop_watch_ptr_->toc("processing/inner", true));

  if (!inference()) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_bevfusion"), "Fail to inference and skip to detect.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/inference_ms", stop_watch_ptr_->toc("processing/inner", true));

  if (!postprocess(det_boxes3d)) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_bevfusion"), "Fail to postprocess and skip to detect.");
    return false;
  }
  proc_timing.emplace(
    "debug/processing_time/postprocess_ms", stop_watch_ptr_->toc("processing/inner", true));

  return true;
}

void BEVFusionTRT::setIntrinsicsExtrinsics(
  std::vector<sensor_msgs::msg::CameraInfo> & camera_info_vector,
  std::vector<Matrix4fRowM> & lidar2camera_vector)
{
  Matrix4fRowM img_aug_matrix = Matrix4fRowM::Identity();
  img_aug_matrix(0, 0) = config_.img_aug_scale_x_;
  img_aug_matrix(1, 1) = config_.img_aug_scale_y_;
  img_aug_matrix(1, 3) = config_.img_aug_offset_y_;

  auto [lidar2images_flattened, geom_feats, kept, ranks, indices] =
    precompute_features(lidar2camera_vector, camera_info_vector, config_);

  assert(static_cast<std::size_t>(lidar2images_flattened.size()) == config_.num_cameras_ * 4 * 4);

  assert(
    static_cast<std::size_t>(geom_feats.size()) <=
    config_.num_cameras_ * 4 * config_.features_height_ * config_.features_width_ *
      config_.num_depth_features_);
  assert(
    static_cast<std::size_t>(kept.size()) == config_.num_cameras_ * config_.features_height_ *
                                               config_.features_width_ *
                                               config_.num_depth_features_);
  assert(
    static_cast<std::size_t>(ranks.size()) <= config_.num_cameras_ * config_.features_height_ *
                                                config_.features_width_ *
                                                config_.num_depth_features_);
  assert(
    static_cast<std::size_t>(indices.size()) <= config_.num_cameras_ * config_.features_height_ *
                                                  config_.features_width_ *
                                                  config_.num_depth_features_);

  num_geom_feats_ = static_cast<std::int64_t>(geom_feats.size());
  num_kept_ = static_cast<std::int64_t>(kept.size());
  num_ranks_ = static_cast<std::int64_t>(ranks.size());
  num_indices_ = static_cast<std::int64_t>(indices.size());

  assert(num_geom_feats_ == 4 * num_ranks_);
  assert(num_ranks_ == num_indices_);

  cudaMemcpy(
    lidar2image_d_.get(), lidar2images_flattened.data(),
    config_.num_cameras_ * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(
    geom_feats_d_.get(), geom_feats.data(), num_geom_feats_ * sizeof(std::int32_t),
    cudaMemcpyHostToDevice);
  cudaMemcpy(kept_d_.get(), kept.data(), num_kept_ * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    ranks_d_.get(), ranks.data(), num_ranks_ * sizeof(std::int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(
    indices_d_.get(), indices.data(), num_indices_ * sizeof(std::int64_t), cudaMemcpyHostToDevice);
}

bool BEVFusionTRT::preprocess(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pc_msg,
  const std::vector<sensor_msgs::msg::Image::ConstSharedPtr> & image_msgs,
  const tf2_ros::Buffer & tf_buffer)
{
  if (!vg_ptr_->enqueuePointCloud(*pc_msg, tf_buffer)) {
    return false;
  }

  // TODO(knzo25): these should be able to be removed as they are filled by TensorRT
  cuda::clear_async(label_pred_output_d_.get(), config_.num_proposals_, stream_);
  cuda::clear_async(bbox_pred_output_d_.get(), bbox_pred_size_, stream_);
  cuda::clear_async(score_output_d_.get(), config_.num_proposals_, stream_);

  cuda::clear_async(voxel_features_d_.get(), voxel_features_size_, stream_);
  cuda::clear_async(voxel_coords_d_.get(), voxel_coords_size_, stream_);
  cuda::clear_async(num_points_per_voxel_d_.get(), config_.max_voxels_, stream_);
  cuda::clear_async(
    points_d_.get(), config_.cloud_capacity_ * config_.num_point_feature_size_, stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  int start_y = config_.resized_height_ - config_.roi_height_;
  int start_x =
    std::max(0, static_cast<int>(config_.resized_width_) - static_cast<int>(config_.roi_width_)) /
    2;

  for (std::size_t camera_id = 0; camera_id < config_.num_cameras_; camera_id++) {
    cudaMemcpyAsync(
      image_buffers_d_[camera_id].get(), image_msgs[camera_id]->data.data(),
      config_.raw_image_height_ * config_.raw_image_width_ * 3, cudaMemcpyHostToDevice, stream_);

    pre_ptr_->resize_and_extract_roi_launch(
      image_buffers_d_[camera_id].get(),
      &roi_tensor_d_[camera_id * config_.roi_height_ * config_.roi_width_ * 3],
      config_.raw_image_height_, config_.raw_image_width_, config_.resized_height_,
      config_.resized_width_, config_.roi_height_, config_.roi_width_, start_y, start_x);
  }

  const auto num_points = vg_ptr_->generateSweepPoints(*pc_msg, points_d_);

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  RCLCPP_INFO_STREAM(
    rclcpp::get_logger("lidar_bevfusion"), "Generated sweep points: " << num_points);

  std::size_t num_voxels = pre_ptr_->generateVoxels(
    points_d_.get(), num_points, voxel_features_d_.get(), voxel_coords_d_.get(), num_points_per_voxel_d_.get());
  // unsigned int params_input;
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  if (num_voxels < config_.min_voxel_size_) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_bevfusion"),
      "Too few voxels (" << num_voxels << ") for the actual optimization profile ("
                         << config_.min_voxel_size_ << ")");
    return false;
  }
  if (num_voxels > config_.max_voxel_size_) {
    RCLCPP_ERROR_STREAM(
      rclcpp::get_logger("lidar_bevfusion"),
      "Actual number of voxels (" << num_voxels
                                  << ") is over the limit for the actual optimization profile ("
                                  << config_.max_voxel_size_ << "). Clipping to the limit.");
    num_voxels = config_.max_voxel_size_;
  }

  RCLCPP_DEBUG_STREAM(
    rclcpp::get_logger("lidar_bevfusion"), "Generated input voxels: " << num_voxels);

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
  input_imgs_shape.d[0] = config_.num_cameras_;
  input_imgs_shape.d[1] = 3;
  input_imgs_shape.d[2] = config_.roi_height_;
  input_imgs_shape.d[3] = config_.roi_width_;

  nvinfer1::Dims input_lidar2image_shape;
  input_lidar2image_shape.nbDims = 3;
  input_lidar2image_shape.d[0] = config_.num_cameras_;
  input_lidar2image_shape.d[1] = 4;
  input_lidar2image_shape.d[2] = 4;

  nvinfer1::Dims input_geom_feats_shape;
  input_geom_feats_shape.nbDims = 2;
  input_geom_feats_shape.d[0] = num_ranks_;
  input_geom_feats_shape.d[1] = 4;

  nvinfer1::Dims input_kept_shape;
  input_kept_shape.nbDims = 1;
  input_kept_shape.d[0] = num_kept_;

  nvinfer1::Dims input_ranks_shape;
  input_ranks_shape.nbDims = 1;
  input_ranks_shape.d[0] = num_ranks_;

  nvinfer1::Dims input_indices_shape;
  input_indices_shape.nbDims = 1;
  input_indices_shape.d[0] = num_indices_;

  std::vector<float> lidar2image_host(config_.num_cameras_ * 4 * 4);
  cudaMemcpy(
    lidar2image_host.data(), lidar2image_d_.get(), config_.num_cameras_ * 4 * 4 * sizeof(float),
    cudaMemcpyDeviceToHost);

  network_trt_ptr_->context->setInputShape("voxels", input_voxels_shape);
  network_trt_ptr_->context->setInputShape("coors", input_coors_shape);
  network_trt_ptr_->context->setInputShape("imgs", input_imgs_shape);
  network_trt_ptr_->context->setInputShape("lidar2image", input_lidar2image_shape);
  network_trt_ptr_->context->setInputShape("geom_feats", input_geom_feats_shape);
  network_trt_ptr_->context->setInputShape("kept", input_kept_shape);
  network_trt_ptr_->context->setInputShape("ranks", input_ranks_shape);
  network_trt_ptr_->context->setInputShape("indices", input_indices_shape);

  network_trt_ptr_->context->setTensorAddress("voxels", voxel_features_d_.get());
  network_trt_ptr_->context->setTensorAddress("coors", voxel_coords_d_.get());
  network_trt_ptr_->context->setTensorAddress("imgs", roi_tensor_d_.get());
  network_trt_ptr_->context->setTensorAddress("lidar2image", lidar2image_d_.get());
  network_trt_ptr_->context->setTensorAddress("geom_feats", geom_feats_d_.get());
  network_trt_ptr_->context->setTensorAddress("kept", kept_d_.get());
  network_trt_ptr_->context->setTensorAddress("ranks", ranks_d_.get());
  network_trt_ptr_->context->setTensorAddress("indices", indices_d_.get());

  network_trt_ptr_->context->setTensorAddress("label_pred", label_pred_output_d_.get());
  network_trt_ptr_->context->setTensorAddress("bbox_pred", bbox_pred_output_d_.get());
  network_trt_ptr_->context->setTensorAddress("score", score_output_d_.get());
  return true;
}

bool BEVFusionTRT::inference()
{
  auto status = network_trt_ptr_->context->enqueueV3(stream_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  if (!status) {
    RCLCPP_WARN_STREAM(
      rclcpp::get_logger("lidar_bevfusion"), "Fail to enqueue and skip to detect.");
    return false;
  }

  return true;
}

bool BEVFusionTRT::postprocess(std::vector<Box3D> & det_boxes3d)
{
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  CHECK_CUDA_ERROR(post_ptr_->generateDetectedBoxes3D_launch(
    label_pred_output_d_.get(), bbox_pred_output_d_.get(), score_output_d_.get(), det_boxes3d,
    stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
  return true;
}

}  //  namespace autoware::lidar_bevfusion
