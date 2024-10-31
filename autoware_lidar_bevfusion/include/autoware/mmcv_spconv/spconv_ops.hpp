// Copyright 2019 Yan Yan
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

#pragma once

#include "autoware/mmcv_spconv/common/pytorch_cpp_helper.hpp"
#include "autoware/mmcv_spconv/common/pytorch_device_registry.hpp"

#include <cstdint>

template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsForwardCUDAKernelLauncher(
    torch::Tensor indices, int64_t batchSize,
    std::vector<std::int64_t> outSpatialShape, std::vector<std::int64_t> spatialShape,
    std::vector<std::int64_t> kernelSize, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation,
    std::vector<std::int64_t> outPadding, int64_t _subM, int64_t _transpose);

/* 
template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher(
    torch::Tensor indices, int64_t batchSize,
    std::vector<std::int64_t> outSpatialShape, std::vector<std::int64_t> spatialShape,
    std::vector<std::int64_t> kernelSize, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation,
    std::vector<std::int64_t> outPadding, int64_t _subM, int64_t _transpose); */
/* 
template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsBackwardCUDAKernelLauncher(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<std::int64_t> outSpatialShape, std::vector<std::int64_t> spatialShape,
    std::vector<std::int64_t> kernelSize, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation,
    std::vector<std::int64_t> outPadding, int64_t _subM, int64_t _transpose); */



template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward_cuda(
    torch::Tensor indices, int64_t batchSize,
    std::vector<std::int64_t> outSpatialShape, std::vector<std::int64_t> spatialShape,
    std::vector<std::int64_t> kernelSize, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation,
    std::vector<std::int64_t> outPadding, int64_t _subM, int64_t _transpose);

/* 
template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward_mlu(
    torch::Tensor indices, int64_t batchSize,
    std::vector<std::int64_t> outSpatialShape, std::vector<std::int64_t> spatialShape,
    std::vector<std::int64_t> kernelSize, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation,
    std::vector<std::int64_t> outPadding, int64_t _subM, int64_t _transpose); */


/* 
template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_backward_cuda(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<std::int64_t> outSpatialShape, std::vector<std::int64_t> spatialShape,
    std::vector<std::int64_t> kernelSize, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation,
    std::vector<std::int64_t> outPadding, int64_t _subM, int64_t _transpose); */



template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward(
    torch::Tensor indices, int64_t batchSize,
    std::vector<std::int64_t> outSpatialShape, std::vector<std::int64_t> spatialShape,
    std::vector<std::int64_t> kernelSize, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation,
    std::vector<std::int64_t> outPadding, int64_t _subM, int64_t _transpose);

/* template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_backward(
    torch::Tensor indices, torch::Tensor gridOut, int64_t batchSize,
    std::vector<std::int64_t> outSpatialShape, std::vector<std::int64_t> spatialShape,
    std::vector<std::int64_t> kernelSize, std::vector<std::int64_t> stride,
    std::vector<std::int64_t> padding, std::vector<std::int64_t> dilation,
    std::vector<std::int64_t> outPadding, int64_t _subM, int64_t _transpose); */

torch::Tensor indice_conv_forward_impl(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM);

torch::Tensor indice_conv_forward(torch::Tensor features, torch::Tensor filters,
                                  torch::Tensor indicePairs,
                                  torch::Tensor indiceNum, int64_t numActOut,
                                  int64_t _inverse, int64_t _subM);

std::vector<torch::Tensor> indice_conv_backward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);

std::vector<torch::Tensor> indice_conv_backward(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);