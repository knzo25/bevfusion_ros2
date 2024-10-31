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

#ifndef AUTOWARE__TENSORRT_PLUGINS__SPARSE_CONV_PLUGIN_HPP_
#define AUTOWARE__TENSORRT_PLUGINS__SPARSE_CONV_PLUGIN_HPP_

#include "autoware/tensorrt_plugins/indice_conv_plugin.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

constexpr char const * const kSPARSE_CONV_PLUGIN_NAME{"SparseConv"};
constexpr char const * const kSPARSE_CONV_PLUGIN_VERSION{"1"};
constexpr char const * const kSPARSE_CONV_PLUGIN_NAMESPACE{""};

namespace nvinfer1
{
namespace plugin
{

class SparseConvPlugin : public IndiceConvPlugin
{
public:
  explicit SparseConvPlugin(const std::string & name);

  ~SparseConvPlugin() override = default;

  // IPluginV3 Methods

  IPluginV3 * clone() noexcept override;

  // IPluginV3OneCore Methods

  char const * getPluginName() const noexcept override;

  char const * getPluginVersion() const noexcept override;

  char const * getPluginNamespace() const noexcept override;

  // IPluginV3OneRuntime Methods

  std::int32_t enqueue(
    PluginTensorDesc const * input_desc, PluginTensorDesc const * output_desc,
    void const * const * inputs, void * const * outputs, void * workspace,
    cudaStream_t stream) noexcept override;
};

}  // namespace plugin
}  // namespace nvinfer1

#endif  // AUTOWARE__TENSORRT_PLUGINS__SPARSE_CONV_PLUGIN_HPP_
