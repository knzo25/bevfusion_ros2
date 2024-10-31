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

#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntime.h>

#include <cstdint>
#include <cstring>
#include <sstream>

void caughtError(std::exception const & e)
{
  getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());
}

void logDebug(char const * msg)
{
  getLogger()->log(nvinfer1::ILogger::Severity::kVERBOSE, msg);
}

void logInfo(char const * msg)
{
  getLogger()->log(nvinfer1::ILogger::Severity::kINFO, msg);
}

void reportAssertion(bool success, char const * msg, char const * file, std::int32_t line)
{
  if (!success) {
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << std::endl
           << file << ':' << line << std::endl
           << "Aborting..." << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    std::abort();
  }
}

void reportValidation(bool success, char const * msg, char const * file, std::int32_t line)
{
  if (!success) {
    std::ostringstream stream;
    stream << "Validation failed: " << msg << std::endl << file << ':' << line << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
  }
}

torch::Tensor toTensor(nvinfer1::PluginTensorDesc desc, void * data)
{
  auto get_shape = [](nvinfer1::PluginTensorDesc desc) {
    std::vector<std::int64_t> shape;

    for (std::int32_t dim = 0; dim < desc.dims.nbDims; dim++) {
      shape.push_back(desc.dims.d[dim]);
    }

    return shape;
  };

  auto shape = get_shape(desc);
  c10::TensorOptions options;

  if (desc.type == nvinfer1::DataType::kFLOAT) {
    options = at::TensorOptions().dtype(torch::kFloat).device(at::kCUDA);
  } else if (desc.type == nvinfer1::DataType::kINT32) {
    options = at::TensorOptions().dtype(torch::kInt32).device(at::kCUDA);
  } else {
    assert(false);
  }

  return torch::from_blob(data, shape, options);
}

torch::Tensor toConstTensor(nvinfer1::PluginTensorDesc desc, void const * data)
{
  auto get_shape = [](nvinfer1::PluginTensorDesc desc) {
    std::vector<std::int64_t> shape;

    for (std::int32_t dim = 0; dim < desc.dims.nbDims; dim++) {
      shape.push_back(desc.dims.d[dim]);
    }

    return shape;
  };

  auto shape = get_shape(desc);
  c10::TensorOptions options;

  if (desc.type == nvinfer1::DataType::kFLOAT) {
    options = at::TensorOptions().dtype(torch::kFloat).device(at::kCUDA);
  } else if (desc.type == nvinfer1::DataType::kINT32) {
    options = at::TensorOptions().dtype(torch::kInt32).device(at::kCUDA);
  } else {
    assert(false);
  }

  // note(knzo25): Shame on me
  torch::Tensor tensor = torch::from_blob(const_cast<void *>(data), shape, options);

  tensor.set_requires_grad(false);   // Optional, ensures no gradients are computed
  tensor = tensor.clone().detach();  // Ensures it is detached from the computational graph

  return tensor;
}