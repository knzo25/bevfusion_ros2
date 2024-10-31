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

#ifndef AUTOWARE__LIDAR_BEVFUSION__PREPROCESS__PREPROCESS_KERNEL_HPP_
#define AUTOWARE__LIDAR_BEVFUSION__PREPROCESS__PREPROCESS_KERNEL_HPP_

#include "autoware/lidar_bevfusion/bevfusion_config.hpp"
#include "autoware/lidar_bevfusion/cuda_utils.hpp"
#include "autoware/lidar_bevfusion/utils.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace autoware::lidar_bevfusion
{

class PreprocessCuda
{
public:
  PreprocessCuda(const BEVFusionConfig & config, cudaStream_t & stream, bool allocate_buffers);

  void generateVoxels(
    float * points, unsigned int points_size, unsigned int * pillar_num, float * voxel_features,
    unsigned int * voxel_num, unsigned int * voxel_idxs);

  cudaError_t generateSweepPoints_launch(
    const std::uint8_t * input_data, std::size_t points_size, int input_point_step, float time_lag,
    const float * transform, float * output_points);

  cudaError_t generateVoxels_random_launch(
    float * points, unsigned int points_size, unsigned int * mask, float * voxels);

  cudaError_t generateBaseFeatures_launch(
    unsigned int * mask, float * voxels, float * voxel_features, std::int32_t * voxel_coordinates,
    unsigned int * voxel_num);

  std::size_t generateVoxels(
    float * points, unsigned int points_size, float * voxel_features, std::int32_t * voxel_coords);

  cudaError_t resize_and_extract_roi_launch(
    const std::uint8_t * input_img, std::uint8_t * output_img, int H, int W, int H2, int W2, int H3,
    int W3, int y_start, int x_start);

private:
  BEVFusionConfig config_;
  cudaStream_t stream_;
  cuda::unique_ptr<unsigned int[]> mask_{nullptr};
  cuda::unique_ptr<float[]> voxels_{nullptr};
  unsigned int mask_size_{};
  unsigned int voxels_size_{};
  cuda::unique_ptr<unsigned int[]> num_voxels_{nullptr};
};
}  // namespace autoware::lidar_bevfusion

#endif  // AUTOWARE__LIDAR_BEVFUSION__PREPROCESS__PREPROCESS_KERNEL_HPP_
