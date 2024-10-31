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

#include "autoware/tensorrt_plugins/sparse_conv_plugin_creator.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"
#include "autoware/tensorrt_plugins/sparse_conv_plugin.hpp"

#include <NvInferRuntimePlugin.h>

#include <exception>
#include <iostream>
#include <mutex>

namespace nvinfer1
{
namespace plugin
{

REGISTER_TENSORRT_PLUGIN(SparseConvPluginCreator);

SparseConvPluginCreator::SparseConvPluginCreator()
{
  fc_.nbFields = 0;
}

nvinfer1::PluginFieldCollection const * SparseConvPluginCreator::getFieldNames() noexcept
{
  return &fc_;
}

IPluginV3 * SparseConvPluginCreator::createPlugin(
  char const * name, PluginFieldCollection const * fc, TensorRTPhase phase) noexcept
{
  PLUGIN_VALIDATE(fc->nbFields == 0);

  if (phase == TensorRTPhase::kBUILD || phase == TensorRTPhase::kRUNTIME) {
    try {
      SparseConvPlugin * const plugin{new SparseConvPlugin{std::string{name}}};
      return plugin;
    } catch (std::exception const & e) {
      caughtError(e);
    }
    return nullptr;
  } else {
    return nullptr;
  }
}

}  // namespace plugin
}  // namespace nvinfer1
