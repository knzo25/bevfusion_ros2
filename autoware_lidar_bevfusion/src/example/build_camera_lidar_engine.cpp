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

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

class CustomLogger : public nvinfer1::ILogger
{
  void log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept override
  {
    if (severity <= nvinfer1::ILogger::Severity::kVERBOSE) {
      std::cout << msg << std::endl;
    }
  }
};

struct InferDeleter
{
  template <typename T>
  void operator()(T * obj) const
  {
    delete obj;
  }
};

int main(int argc, char ** argv)
{
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <onnx_file_path> <plugin_library_path> <engine_file_path>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string const onnx_file_path{argv[1]};
  std::string const plugin_library_path{argv[2]};
  std::string const engine_file_path{argv[3]};

  std::cout << "ONNX file path: " << onnx_file_path << std::endl;
  std::cout << "Plugin library path: " << plugin_library_path << std::endl;
  std::cout << "Engine file path: " << engine_file_path << std::endl;

  CustomLogger logger{};

  // char const* const plugin_library_path_c_str{plugin_library_path.c_str()};

  // Create the builder.
  std::unique_ptr<nvinfer1::IBuilder, InferDeleter> builder{nvinfer1::createInferBuilder(logger)};

  if (builder == nullptr) {
    std::cerr << "Failed to create the builder." << std::endl;
    return EXIT_FAILURE;
  }

  auto profile = builder->createOptimizationProfile();

  profile->setDimensions("voxels", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 5});
  profile->setDimensions("voxels", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{128000, 5});
  profile->setDimensions("voxels", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{256000, 5});

  profile->setDimensions("coors", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 4});
  profile->setDimensions("coors", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{128000, 4});
  profile->setDimensions("coors", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{256000, 4});

  profile->setDimensions(
    "imgs", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 256, 704});
  profile->setDimensions(
    "imgs", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{6, 3, 256, 704});
  profile->setDimensions(
    "imgs", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{6, 3, 256, 704});

  profile->setDimensions(
    "lidar2image", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{1, 4, 4});
  profile->setDimensions(
    "lidar2image", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3{6, 4, 4});
  profile->setDimensions(
    "lidar2image", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3{6, 4, 4});

  nvinfer1::Dims kept_min_dims, kept_max_dims;
  kept_min_dims.nbDims = 1;
  kept_min_dims.d[0] = 0 * 118 * 32 * 88;
  kept_max_dims.nbDims = 1;
  kept_max_dims.d[0] = 6 * 118 * 32 * 88;

  nvinfer1::Dims geom_feats_min_dims, geom_feats_opt_dims, geom_feats_max_dims;
  geom_feats_min_dims.nbDims = 1;
  geom_feats_min_dims.d[0] = 0 * 118 * 32 * 88;
  geom_feats_opt_dims.nbDims = 1;
  geom_feats_opt_dims.d[0] = 6 * 118 * 32 * 88 / 2;
  geom_feats_max_dims.nbDims = 1;
  geom_feats_max_dims.d[0] = 6 * 118 * 32 * 88;

  profile->setDimensions(
    "geom_feats", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{geom_feats_min_dims.d[0], 4});
  profile->setDimensions(
    "geom_feats", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{geom_feats_opt_dims.d[0], 4});
  profile->setDimensions(
    "geom_feats", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{geom_feats_max_dims.d[0], 4});

  profile->setDimensions("kept", nvinfer1::OptProfileSelector::kMIN, kept_min_dims);
  profile->setDimensions("kept", nvinfer1::OptProfileSelector::kOPT, kept_max_dims);
  profile->setDimensions("kept", nvinfer1::OptProfileSelector::kMAX, kept_max_dims);

  profile->setDimensions("ranks", nvinfer1::OptProfileSelector::kMIN, geom_feats_min_dims);
  profile->setDimensions("ranks", nvinfer1::OptProfileSelector::kOPT, geom_feats_opt_dims);
  profile->setDimensions("ranks", nvinfer1::OptProfileSelector::kMAX, geom_feats_max_dims);

  profile->setDimensions("indices", nvinfer1::OptProfileSelector::kMIN, geom_feats_min_dims);
  profile->setDimensions("indices", nvinfer1::OptProfileSelector::kOPT, geom_feats_opt_dims);
  profile->setDimensions("indices", nvinfer1::OptProfileSelector::kMAX, geom_feats_max_dims);

  void * const plugin_handle{builder->getPluginRegistry().loadLibrary(plugin_library_path.c_str())};
  if (plugin_handle == nullptr) {
    std::cerr << "Failed to load the plugin library." << std::endl;
    return EXIT_FAILURE;
  }

  // Create the network.
  std::uint32_t flag{0U};
  // For TensorRT < 10.0, explicit dimension has to be specified to
  // distinguish from the implicit dimension. For TensorRT >= 10.0, explicit
  // dimension is the only choice and this flag has been deprecated.
  if (getInferLibVersion() < 100000) {
    flag |=
      1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  }
  std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> network{
    builder->createNetworkV2(flag)};
  if (network == nullptr) {
    std::cerr << "Failed to create the network." << std::endl;
    return EXIT_FAILURE;
  }

  // Create the parser.
  std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser{
    nvonnxparser::createParser(*network, logger)};
  if (parser == nullptr) {
    std::cerr << "Failed to create the parser." << std::endl;
    return EXIT_FAILURE;
  }
  parser->parseFromFile(
    onnx_file_path.c_str(), static_cast<std::int32_t>(nvinfer1::ILogger::Severity::kWARNING));
  for (std::int32_t i = 0; i < parser->getNbErrors(); ++i) {
    std::cout << parser->getError(i)->desc() << std::endl;
  }

  // Set the allowed IO tensor formats.
  std::uint32_t const formats{1U << static_cast<std::uint32_t>(nvinfer1::TensorFormat::kLINEAR)};
  nvinfer1::DataType const dtype{nvinfer1::DataType::kFLOAT};
  network->getInput(0)->setAllowedFormats(formats);
  network->getInput(0)->setType(dtype);
  network->getOutput(0)->setAllowedFormats(formats);
  network->getOutput(0)->setType(dtype);

  // Build the engine.
  std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter> config{builder->createBuilderConfig()};
  if (config == nullptr) {
    std::cerr << "Failed to create the builder config." << std::endl;
    return EXIT_FAILURE;
  }

  // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 33);
  // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);
  // config->setMaxWorkspaceSize(1 << 33);
  // config->setBuilderOptimizationLevel(0); //default is 3
  // config->setMaxNbTactics(1);

  // config->setFlag(nvinfer1::BuilderFlag::kFP16); this can force the output to be half
  config->addOptimizationProfile(profile);

  std::unique_ptr<nvinfer1::IHostMemory, InferDeleter> serializedModel{
    builder->buildSerializedNetwork(*network, *config)};

  // Write the serialized engine to a file.
  std::ofstream engineFile{engine_file_path.c_str(), std::ios::binary};
  if (!engineFile.is_open()) {
    std::cerr << "Failed to open the engine file." << std::endl;
    return EXIT_FAILURE;
  }
  engineFile.write(static_cast<char const *>(serializedModel->data()), serializedModel->size());
  engineFile.close();

  std::cout << "Successfully serialized the engine to the file: " << engine_file_path << std::endl;

  return EXIT_SUCCESS;
}
