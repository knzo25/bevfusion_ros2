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
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "autoware/lidar_bevfusion/cuda_utils.hpp"
#include "autoware/lidar_bevfusion/preprocess/preprocess_kernel.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace autoware::lidar_bevfusion
{

PreprocessCuda::PreprocessCuda(
  const BEVFusionConfig & config, cudaStream_t & stream, bool allocate_buffers)
: stream_(stream), config_(config)
{
  mask_size_ = config_.grid_z_size_ * config_.grid_y_size_ * config_.grid_x_size_;
  voxels_size_ = config_.grid_z_size_ * config_.grid_y_size_ * config_.grid_x_size_ * 5;

  if (allocate_buffers) {
    mask_ = cuda::make_unique<unsigned int[]>(mask_size_);
    voxels_ = cuda::make_unique<float[]>(voxels_size_);
    num_voxels_ = cuda::make_unique<unsigned int[]>(1);
    ;
  }
}

__global__ void generateSweepPoints_kernel(
  const std::uint8_t * input_data, std::size_t points_size, int input_point_step, float time_lag,
  const float * transform_array, int num_features, float * output_points)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) return;

  union {
    std::uint32_t raw{0};
    float value;
  } input_x, input_y, input_z;

#pragma unroll
  for (int i = 0; i < 4; i++) {  // 4 bytes for float32
    input_x.raw |= input_data[point_idx * input_point_step + i] << i * 8;
    input_y.raw |= input_data[point_idx * input_point_step + i + 4] << i * 8;
    input_z.raw |= input_data[point_idx * input_point_step + i + 8] << i * 8;
  }

  float input_intensity = static_cast<float>(input_data[point_idx * input_point_step + 12]);

  output_points[point_idx * num_features] =
    transform_array[0] * input_x.value + transform_array[4] * input_y.value +
    transform_array[8] * input_z.value + transform_array[12];
  output_points[point_idx * num_features + 1] =
    transform_array[1] * input_x.value + transform_array[5] * input_y.value +
    transform_array[9] * input_z.value + transform_array[13];
  output_points[point_idx * num_features + 2] =
    transform_array[2] * input_x.value + transform_array[6] * input_y.value +
    transform_array[10] * input_z.value + transform_array[14];
  output_points[point_idx * num_features + 3] = input_intensity;
  output_points[point_idx * num_features + 4] = time_lag;
}

cudaError_t PreprocessCuda::generateSweepPoints_launch(
  const std::uint8_t * input_data, std::size_t points_size, int input_point_step, float time_lag,
  const float * transform_array, float * output_points)
{
  dim3 blocks(divup(points_size, config_.threads_for_voxel_));
  dim3 threads(config_.threads_for_voxel_);

  generateSweepPoints_kernel<<<blocks, threads, 0, stream_>>>(
    input_data, points_size, input_point_step, time_lag, transform_array,
    config_.num_point_feature_size_, output_points);

  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void generateVoxels_random_kernel(
  float * points, unsigned int points_size, float min_x_range, float max_x_range, float min_y_range,
  float max_y_range, float min_z_range, float max_z_range, float voxel_x_size, float voxel_y_size,
  float voxel_z_size, int grid_y_size, int grid_x_size, unsigned int * mask, float * voxels)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_idx >= points_size) return;

  float x = points[point_idx * 5];
  float y = points[point_idx * 5 + 1];
  float z = points[point_idx * 5 + 2];
  float i = points[point_idx * 5 + 3];
  float t = points[point_idx * 5 + 4];

  if (
    x <= min_x_range || x >= max_x_range || y <= min_y_range || y >= max_y_range ||
    z <= min_z_range || z >= max_z_range)
    return;

  int voxel_idx = floorf((x - min_x_range) / voxel_x_size);
  int voxel_idy = floorf((y - min_y_range) / voxel_y_size);
  int voxel_idz = floorf((z - min_z_range) / voxel_z_size);
  unsigned int voxel_index =
    voxel_idz * grid_x_size * grid_y_size + voxel_idy * grid_x_size + voxel_idx;

  unsigned int point_id = atomicAdd(&(mask[voxel_index]), 1);

  float * address = voxels + 5 * voxel_index;

  atomicAdd(address + 0, x);
  atomicAdd(address + 1, y);
  atomicAdd(address + 2, z);
  atomicAdd(address + 3, i);
  atomicAdd(address + 4, t);
}

cudaError_t PreprocessCuda::generateVoxels_random_launch(
  float * points, unsigned int points_size, unsigned int * mask, float * voxels)
{
  if (points_size == 0) {
    return cudaGetLastError();
  }
  dim3 blocks(divup(points_size, 256));
  dim3 threads(256);

  generateVoxels_random_kernel<<<blocks, threads, 0, stream_>>>(
    points, points_size, config_.min_x_range_, config_.max_x_range_, config_.min_y_range_,
    config_.max_y_range_, config_.min_z_range_, config_.max_z_range_, config_.voxel_x_size_,
    config_.voxel_y_size_, config_.voxel_z_size_, config_.grid_y_size_, config_.grid_x_size_, mask,
    voxels);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void generateBaseFeatures_kernel(
  unsigned int * mask, float * voxels, int grid_z_size, int grid_y_size, int grid_x_size,
  float max_voxels, float * voxel_features, std::int32_t * voxel_coords, unsigned int * voxel_num)
{
  unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int voxel_idy = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int voxel_idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (voxel_idx >= grid_x_size || voxel_idy >= grid_y_size || voxel_idz >= grid_z_size) return;

  unsigned int input_voxel_index =
    voxel_idz * grid_y_size * grid_x_size + voxel_idy * grid_x_size + voxel_idx;
  unsigned int count = mask[input_voxel_index];
  if (!(count > 0)) return;

  // unsigned int current_pillarId = 0;
  unsigned int output_voxel_index = atomicAdd(voxel_num, 1);

  voxel_features[5 * output_voxel_index + 0] = voxels[5 * input_voxel_index + 0] / count;
  voxel_features[5 * output_voxel_index + 1] = voxels[5 * input_voxel_index + 1] / count;
  voxel_features[5 * output_voxel_index + 2] = voxels[5 * input_voxel_index + 2] / count;
  voxel_features[5 * output_voxel_index + 3] = voxels[5 * input_voxel_index + 3] / count;
  voxel_features[5 * output_voxel_index + 4] = voxels[5 * input_voxel_index + 4] / count;

  voxel_coords[4 * output_voxel_index + 0] = 0;
  voxel_coords[4 * output_voxel_index + 1] = voxel_idy;
  voxel_coords[4 * output_voxel_index + 2] = voxel_idx;
  voxel_coords[4 * output_voxel_index + 3] = voxel_idz;
}

// create 4 channels
cudaError_t PreprocessCuda::generateBaseFeatures_launch(
  unsigned int * mask, float * voxels, float * voxel_features, std::int32_t * voxel_coordinates,
  unsigned int * voxel_num)
{
  dim3 threads = {16, 16, 4};
  dim3 blocks = {
    divup(config_.grid_x_size_, threads.x), divup(config_.grid_y_size_, threads.y),
    divup(config_.grid_z_size_, threads.z)};

  generateBaseFeatures_kernel<<<blocks, threads, 0, stream_>>>(
    mask, voxels, config_.grid_z_size_, config_.grid_y_size_, config_.grid_x_size_,
    config_.max_voxels_, voxel_features, voxel_coordinates, voxel_num);
  cudaError_t err = cudaGetLastError();
  return err;
}

std::size_t PreprocessCuda::generateVoxels(
  float * points, unsigned int points_size, float * voxel_features, std::int32_t * voxel_coords)
{
  cudaMemsetAsync(mask_.get(), 0, mask_size_ * sizeof(unsigned int), stream_);
  cudaMemsetAsync(voxels_.get(), 0, voxels_size_ * sizeof(float), stream_);
  cudaMemsetAsync(num_voxels_.get(), 0, 1 * sizeof(unsigned int), stream_);

  CHECK_CUDA_ERROR(generateVoxels_random_launch(points, points_size, mask_.get(), voxels_.get()));

  CHECK_CUDA_ERROR(generateBaseFeatures_launch(
    mask_.get(), voxels_.get(), voxel_features, voxel_coords, num_voxels_.get()));

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  unsigned int num_voxels_host;
  CHECK_CUDA_ERROR(
    cudaMemcpy(&num_voxels_host, num_voxels_.get(), sizeof(unsigned int), cudaMemcpyDeviceToHost));

  return static_cast<std::size_t>(num_voxels_host);
}

__global__ void resize_and_extract_roi(
  const std::uint8_t * __restrict__ input_img, std::uint8_t * __restrict__ output_img, int H,
  int W,                     // Original image dimensions (Height, Width)
  int H2, int W2,            // Resized image dimensions (Height, Width)
  int H3, int W3,            // ROI dimensions (Height, Width)
  int y_start, int x_start)  // ROI top-left coordinates in resized image
{
  // Calculate the global thread indices
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // Width index in output (ROI) image
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // Height index in output (ROI) image

  // Check if the thread corresponds to a valid pixel in the ROI
  if (i >= W3 || j >= H3) return;

  // Compute scaling factors from original to resized image
  float scale_y = static_cast<float>(H) / H2;
  float scale_x = static_cast<float>(W) / W2;

  // Map output pixel (i, j) in ROI to position in resized image
  int i_resized = i + x_start;
  int j_resized = j + y_start;

  // Map position in resized image back to position in original image
  float i_original = (i_resized + 0.5f) * scale_x - 0.5f;
  float j_original = (j_resized + 0.5f) * scale_y - 0.5f;

  // Compute coordinates for bilinear interpolation
  int i0 = static_cast<int>(floorf(i_original));
  int j0 = static_cast<int>(floorf(j_original));
  int i1 = i0 + 1;
  int j1 = j0 + 1;

  // Compute interpolation weights
  float di = i_original - i0;
  float dj = j_original - j0;

  float w00 = (1.0f - di) * (1.0f - dj);
  float w01 = (1.0f - di) * dj;
  float w10 = di * (1.0f - dj);
  float w11 = di * dj;

  // Loop over the three color channels
  for (int c = 0; c < 3; ++c) {
    float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;

    // Boundary checks for each neighboring pixel
    if (i0 >= 0 && i0 < W && j0 >= 0 && j0 < H)
      v00 = static_cast<float>(input_img[(j0 * W + i0) * 3 + c]);
    if (i0 >= 0 && i0 < W && j1 >= 0 && j1 < H)
      v01 = static_cast<float>(input_img[(j1 * W + i0) * 3 + c]);
    if (i1 >= 0 && i1 < W && j0 >= 0 && j0 < H)
      v10 = static_cast<float>(input_img[(j0 * W + i1) * 3 + c]);
    if (i1 >= 0 && i1 < W && j1 >= 0 && j1 < H)
      v11 = static_cast<float>(input_img[(j1 * W + i1) * 3 + c]);

    // Compute the interpolated pixel value
    float value = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;

    // Store the result in the output ROI image
    output_img[(j * W3 + i) * 3 + c] = static_cast<std::uint8_t>(value);
  }
}

cudaError_t PreprocessCuda::resize_and_extract_roi_launch(
  const std::uint8_t * input_img, std::uint8_t * output_img, int H,
  int W,                     // Original image dimensions
  int H2, int W2,            // Resized image dimensions
  int H3, int W3,            // ROI dimensions
  int y_start, int x_start)  // ROI top-left coordinates in resized image
{
  // Define the block and grid dimensions
  dim3 threads(16, 16);
  dim3 blocks(divup(W3, threads.x), divup(H3, threads.y));

  // Launch the kernel
  resize_and_extract_roi<<<blocks, threads, 0, stream_>>>(
    input_img, output_img, H, W, H2, W2, H3, W3, y_start, x_start);

  // Check for errors
  return cudaGetLastError();
}

}  // namespace autoware::lidar_bevfusion
