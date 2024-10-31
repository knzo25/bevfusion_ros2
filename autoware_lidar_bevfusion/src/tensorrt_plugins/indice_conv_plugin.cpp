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

#include "autoware/tensorrt_plugins/indice_conv_plugin.hpp"

#include "autoware/mmcv_spconv/common/pytorch_device_registry.hpp"
#include "autoware/mmcv_spconv/spconv_ops.hpp"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <torch/torch.h>

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

IndiceConvPlugin::IndiceConvPlugin(const std::string & name) : layer_name_{name}
{
  initFieldsToSerialize();
}

void IndiceConvPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * IndiceConvPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

std::int32_t IndiceConvPlugin::getNbOutputs() const noexcept
{
  return 1;
}

std::int32_t IndiceConvPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(in[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(in[1].desc.dims.nbDims == 4 || in[1].desc.dims.nbDims == 5);
  PLUGIN_ASSERT(in[2].desc.dims.nbDims == 3);
  PLUGIN_ASSERT(in[3].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(in[4].desc.dims.nbDims == 2);

  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 2);

  // input features
  auto num_input_features = in[0].desc.dims.d[1];
  auto num_output_features = out[0].desc.dims.d[1];
  auto num_filter_dims = in[1].desc.dims.nbDims;

  PLUGIN_ASSERT(in[0].desc.dims.d[0] == out[0].desc.dims.d[0]);
  PLUGIN_ASSERT(num_input_features == in[1].desc.dims.d[num_filter_dims - 2]);
  PLUGIN_ASSERT(num_output_features == in[1].desc.dims.d[num_filter_dims - 1]);
  PLUGIN_ASSERT(in[1].desc.type == out[0].desc.type);

  return 0;
}

bool IndiceConvPlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 5 && num_outputs == 1 && pos < num_inputs + num_outputs);
  bool valid = false;

  switch (pos) {
    case 0:  // features
    case 1:  // filters
    case 5:  // output
      valid |=
        (in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         in_out[pos].desc.type == nvinfer1::DataType::kFLOAT);
      break;
    case 2:  // indice_pairs
    case 3:  // indice_pairs_num
    case 4:  // num_active_output
      valid |=
        (in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
         in_out[pos].desc.type == nvinfer1::DataType::kINT32);
      break;

    default:
      break;
  }

  return valid;
}

std::int32_t IndiceConvPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  output_types[0] = input_types[0];
  return 0;
}

std::int32_t IndiceConvPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs,
  [[maybe_unused]] IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(inputs != nullptr);
  PLUGIN_ASSERT(inputs[0].nbDims == 2);  // features
  PLUGIN_ASSERT(
    inputs[1].nbDims == 4 ||
    inputs[1].nbDims == 5);              // weights (ksize x ksize * ksize x input_dim x output_dim)
  PLUGIN_ASSERT(inputs[2].nbDims == 3);  // indice_pairs
  PLUGIN_ASSERT(inputs[3].nbDims == 1);  // indice_pairs_num
  PLUGIN_ASSERT(inputs[4].nbDims == 2);  // num_active_output

  outputs[0].nbDims = 2;
  outputs[0].d[0] = inputs[4].d[0];
  outputs[0].d[1] = inputs[1].d[inputs[1].nbDims - 1];  // the last dim of weights is output_dim

  return 0;
}

std::int32_t IndiceConvPlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * IndiceConvPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * IndiceConvPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t IndiceConvPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return 0;
}

}  // namespace plugin
}  // namespace nvinfer1
