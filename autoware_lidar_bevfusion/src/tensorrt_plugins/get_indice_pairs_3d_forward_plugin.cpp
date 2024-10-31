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

#include "autoware/tensorrt_plugins/get_indice_pairs_3d_forward_plugin.hpp"

#include "autoware/mmcv_spconv/spconv_ops.hpp"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

GetIndicePairs3dForwardPlugin::GetIndicePairs3dForwardPlugin(
  const std::string & name, GetIndicePairs3dForwardParameters const & params)
: layer_name_{name}, params_{params}
{
  initFieldsToSerialize();
}

void GetIndicePairs3dForwardPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  data_to_serialize_.emplace_back("batch_size", &params_.batch_size, PluginFieldType::kINT32, 1);
  data_to_serialize_.emplace_back(
    "dilation_dims", &params_.dilation_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back("ksize_dims", &params_.ksize_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back(
    "out_padding_dims", &params_.out_padding_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back(
    "out_shape_dims", &params_.out_shape_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back("padding_dims", &params_.padding_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back(
    "spatial_shape_dims", &params_.spatial_shape_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back("stride_dims", &params_.stride_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back("subm", &params_.subm, PluginFieldType::kINT32, 1);
  data_to_serialize_.emplace_back("transpose", &params_.transpose, PluginFieldType::kINT32, 1);

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * GetIndicePairs3dForwardPlugin::getCapabilityInterface(
  PluginCapabilityType type) noexcept
{
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild *>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime *>(this);
    }
    PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore *>(this);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV3 * GetIndicePairs3dForwardPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new GetIndicePairs3dForwardPlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * GetIndicePairs3dForwardPlugin::getPluginName() const noexcept
{
  return kGET_INDICE_PAIRS_3D_FORWARD_PLUGIN_NAME;
}

char const * GetIndicePairs3dForwardPlugin::getPluginVersion() const noexcept
{
  return kGET_INDICE_PAIRS_3D_FORWARD_PLUGIN_VERSION;
}

char const * GetIndicePairs3dForwardPlugin::getPluginNamespace() const noexcept
{
  return kGET_INDICE_PAIRS_3D_FORWARD_PLUGIN_NAMESPACE;
}

std::int32_t GetIndicePairs3dForwardPlugin::getNbOutputs() const noexcept
{
  return 4;
}

std::int32_t GetIndicePairs3dForwardPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 4);
  PLUGIN_ASSERT(in[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(out[1].desc.dims.nbDims == 3);
  PLUGIN_ASSERT(out[2].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[3].desc.dims.nbDims == 0);

  std::int64_t kernel_volume = 1;
  for (const std::int64_t ksize : params_.ksize) {
    kernel_volume *= ksize;
  }

  PLUGIN_ASSERT(
    (in[0].desc.dims.d[0] == -1) ||
    (out[0].desc.dims.d[0] == kernel_volume * in[0].desc.dims.d[0]));
  PLUGIN_ASSERT(out[0].desc.dims.d[1] == 4);  // coords + 1
  PLUGIN_ASSERT(out[0].desc.type == in[0].desc.type);

  PLUGIN_ASSERT(out[1].desc.dims.d[0] == kernel_volume);
  PLUGIN_ASSERT(out[1].desc.dims.d[1] == 2);
  PLUGIN_ASSERT(out[1].desc.dims.d[2] == in[0].desc.dims.d[0]);
  PLUGIN_ASSERT(out[1].desc.type == in[0].desc.type);

  PLUGIN_ASSERT(out[2].desc.dims.d[0] == kernel_volume);
  PLUGIN_ASSERT(out[2].desc.type == in[0].desc.type);

  PLUGIN_ASSERT(out[3].desc.type == in[0].desc.type);

  return 0;
}

bool GetIndicePairs3dForwardPlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 4);

  return (
    in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
    in_out[pos].desc.type == nvinfer1::DataType::kINT32);
}

std::int32_t GetIndicePairs3dForwardPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 4);

  output_types[0] = input_types[0];
  output_types[1] = input_types[0];
  output_types[2] = input_types[0];
  output_types[3] = input_types[0];

  return 0;
}

std::int32_t GetIndicePairs3dForwardPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs, IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 4);
  PLUGIN_ASSERT(inputs[0].nbDims == 2);

  std::int64_t kernel_volume = 1;
  std::int64_t input_spatial_volume = 1;
  std::int64_t output_spatial_volume = 1;

  for (std::size_t i = 0; i < params_.ksize.size(); ++i) {
    kernel_volume *= params_.ksize[i];
    input_spatial_volume *= params_.spatial_shape[i];
    output_spatial_volume *= params_.out_shape[i];
  }

  // outids
  if (params_.subm) {
    outputs[0] = inputs[0];
  } else {
    outputs[0].nbDims = 2;
    outputs[0].d[1] = inputs[0].d[1];

    if (params_.spatial_shape[0] > 1000 || 7 * output_spatial_volume > input_spatial_volume) {
      // On average, the number of elements increases, but in practice not so much
      auto upper_bound = expr_builder.operation(
        DimensionOperation::kPROD, *inputs[0].d[0],
        *expr_builder.constant(
          2));  // if we set this to the theoretical limit we run ot ouf memory. a more thorough
                // (empitical) research is needed*expr_builder.constant(volume));
      auto opt_value = inputs[0].d[0];
      auto num_non_zero_size_tensor = expr_builder.declareSizeTensor(3, *opt_value, *upper_bound);
      outputs[0].d[0] = num_non_zero_size_tensor;
    } else {
      // Heuristic: under these conditions the outids halves in size
      auto upper_bound = expr_builder.operation(
        DimensionOperation::kPROD, *inputs[0].d[0],
        *expr_builder.constant(2));  // if we set this to the theoretical limit we run ot ouf
                                     // memory. a more thorough (empitical) research is needed
      auto opt_value = expr_builder.operation(
        DimensionOperation::kCEIL_DIV, *inputs[0].d[0], *expr_builder.constant(2));
      auto num_non_zero_size_tensor = expr_builder.declareSizeTensor(3, *opt_value, *upper_bound);
      outputs[0].d[0] = num_non_zero_size_tensor;
    }
  }

  // indice_pairs
  outputs[1].nbDims = 3;
  outputs[1].d[0] = expr_builder.constant(kernel_volume);
  outputs[1].d[1] = expr_builder.constant(2);
  outputs[1].d[2] = inputs[0].d[0];

  // indice_pair_num
  outputs[2].nbDims = 1;
  outputs[2].d[0] = expr_builder.constant(kernel_volume);

  // num_activate_out
  outputs[3].nbDims = 0;

  return 0;
}

std::int32_t GetIndicePairs3dForwardPlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  torch::Tensor indices = toConstTensor(input_desc[0], inputs[0]);

  // Perform tensor operations on GPU (e.g., pass through a neural network model)
  std::vector<torch::Tensor> output_tensors = get_indice_pairs_forward_cuda<3>(
    indices, params_.batch_size, params_.out_shape, params_.spatial_shape, params_.ksize,
    params_.stride, params_.padding, params_.dilation, params_.out_padding, params_.subm,
    params_.transpose);

  // Copy the output_tensor to the output data (TensorRT expects raw data pointers for outputs)
  std::int32_t * outids_data = static_cast<std::int32_t *>(outputs[0]);
  std::int32_t * indice_pairs_data = static_cast<std::int32_t *>(outputs[1]);
  std::int32_t * indice_num_data = static_cast<std::int32_t *>(outputs[2]);
  std::int32_t * num_act_out_data = static_cast<std::int32_t *>(outputs[3]);

  int numActOut = output_tensors[0].size(0);

  cudaMemcpyAsync(
    outids_data, output_tensors[0].data_ptr<std::int32_t>(),
    output_tensors[0].numel() * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
    indice_pairs_data, output_tensors[1].data_ptr<std::int32_t>(),
    output_tensors[1].numel() * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
    indice_num_data, output_tensors[2].data_ptr<std::int32_t>(),
    output_tensors[2].numel() * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream);
  cudaError_t const status = cudaMemcpyAsync(
    num_act_out_data, &numActOut, sizeof(std::int32_t), cudaMemcpyHostToDevice, stream);

  return status;
}

std::int32_t GetIndicePairs3dForwardPlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * GetIndicePairs3dForwardPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * GetIndicePairs3dForwardPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t GetIndicePairs3dForwardPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return 0;
}

}  // namespace plugin
}  // namespace nvinfer1
