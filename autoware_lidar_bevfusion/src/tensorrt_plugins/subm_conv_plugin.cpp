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

#include "autoware/tensorrt_plugins/subm_conv_plugin.hpp"

#include "autoware/mmcv_spconv/common/pytorch_device_registry.hpp"
#include "autoware/mmcv_spconv/spconv_ops.hpp"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

SubMConvPlugin::SubMConvPlugin(const std::string & name) : IndiceConvPlugin(name)
{
}

IPluginV3 * SubMConvPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new SubMConvPlugin{layer_name_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * SubMConvPlugin::getPluginName() const noexcept
{
  return kSUB_M_CONV_PLUGIN_NAME;
}

char const * SubMConvPlugin::getPluginVersion() const noexcept
{
  return kSUB_M_CONV_PLUGIN_VERSION;
}

char const * SubMConvPlugin::getPluginNamespace() const noexcept
{
  return kSUB_M_CONV_PLUGIN_NAMESPACE;
}

std::int32_t SubMConvPlugin::enqueue(
  PluginTensorDesc const * input_desc, PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  const std::uint32_t num_output_indices = output_desc[0].dims.d[0];

  torch::Tensor features = toConstTensor(input_desc[0], inputs[0]);
  torch::Tensor filters = toConstTensor(input_desc[1], inputs[1]);
  torch::Tensor indice_pairs = toConstTensor(input_desc[2], inputs[2]);
  torch::Tensor indice_num = toConstTensor(input_desc[3], inputs[3]);

  torch::Tensor output_tensor = indice_conv_forward(
    features, filters, indice_pairs, indice_num, num_output_indices, false, true);
  cudaError_t const status = cudaMemcpyAsync(
    static_cast<float *>(outputs[0]), output_tensor.data_ptr<float>(),
    output_tensor.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);

  return status;
}

}  // namespace plugin
}  // namespace nvinfer1
