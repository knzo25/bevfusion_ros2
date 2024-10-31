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

#include "autoware/lidar_bevfusion/network/network_trt.hpp"

#include <NvOnnxParser.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>

namespace autoware::lidar_bevfusion
{

inline NetworkIO nameToNetworkIO(const char * name)
{
  static const std::unordered_map<std::string_view, NetworkIO> name_to_enum = {
    {"voxels", NetworkIO::voxels},        {"num_points", NetworkIO::num_points},
    {"coors", NetworkIO::coors},          {"cls_score0", NetworkIO::cls_score},
    {"bbox_pred0", NetworkIO::bbox_pred}, {"dir_cls_pred0", NetworkIO::dir_pred}};

  auto it = name_to_enum.find(name);
  if (it != name_to_enum.end()) {
    return it->second;
  }
  throw std::runtime_error("Invalid input name: " + std::string(name));
}

std::ostream & operator<<(std::ostream & os, const ProfileDimension & profile)
{
  std::string delim = "";
  os << "min->[";
  for (int i = 0; i < profile.min.nbDims; ++i) {
    os << delim << profile.min.d[i];
    delim = ", ";
  }
  os << "], opt->[";
  delim = "";
  for (int i = 0; i < profile.opt.nbDims; ++i) {
    os << delim << profile.opt.d[i];
    delim = ", ";
  }
  os << "], max->[";
  delim = "";
  for (int i = 0; i < profile.max.nbDims; ++i) {
    os << delim << profile.max.d[i];
    delim = ", ";
  }
  os << "]";
  return os;
}

NetworkTRT::NetworkTRT(const BEVFusionConfig & config) : config_(config)
{
  auto voxels_num_min = static_cast<std::int64_t>(config.voxels_num_[0]);
  auto voxels_num_opt = static_cast<std::int64_t>(config.voxels_num_[1]);
  auto voxels_num_max = static_cast<std::int64_t>(config.voxels_num_[2]);

  nvinfer1::Dims voxels_min_dims = nvinfer1::Dims2{voxels_num_min, 5};
  nvinfer1::Dims voxels_opt_dims = nvinfer1::Dims2{voxels_num_opt, 5};
  nvinfer1::Dims voxels_max_dims = nvinfer1::Dims2{voxels_num_max, 5};

  ProfileDimension voxels_dims = {voxels_min_dims, voxels_opt_dims, voxels_max_dims};
  in_profile_dims_["voxels"] = voxels_dims;

  nvinfer1::Dims coors_min_dims = nvinfer1::Dims2{voxels_num_min, 4};
  nvinfer1::Dims coors_opt_dims = nvinfer1::Dims2{voxels_num_opt, 4};
  nvinfer1::Dims coors_max_dims = nvinfer1::Dims2{voxels_num_max, 4};

  ProfileDimension coors_dims = {coors_min_dims, coors_opt_dims, coors_max_dims};
  in_profile_dims_["coors"] = coors_dims;

  auto roi_height = static_cast<std::int64_t>(config.roi_height_);
  auto roi_width = static_cast<std::int64_t>(config.roi_width_);
  auto num_cameras = static_cast<std::int64_t>(config.num_cameras_);

  nvinfer1::Dims imgs_min_dims = nvinfer1::Dims4{1, 3, roi_height, roi_width};
  nvinfer1::Dims imgs_opt_dims = nvinfer1::Dims4{num_cameras, 3, roi_height, roi_width};
  nvinfer1::Dims imgs_max_dims = nvinfer1::Dims4{num_cameras, 3, roi_height, roi_width};

  ProfileDimension imgs_dims = {imgs_min_dims, imgs_opt_dims, imgs_max_dims};
  in_profile_dims_["imgs"] = imgs_dims;

  nvinfer1::Dims lidar2image_min_dims = nvinfer1::Dims3{1, 4, 4};
  nvinfer1::Dims lidar2image_opt_dims = nvinfer1::Dims3{num_cameras, 4, 4};
  nvinfer1::Dims lidar2image_max_dims = nvinfer1::Dims3{num_cameras, 4, 4};

  ProfileDimension lidar2image_dims = {
    lidar2image_min_dims, lidar2image_opt_dims, lidar2image_max_dims};
  in_profile_dims_["lidar2image"] = lidar2image_dims;

  nvinfer1::Dims kept_min_dims, kept_max_dims;
  std::int64_t feature_tensor_size =
    num_cameras * config.num_depth_features_ * config.features_height_ * config.features_width_;
  kept_min_dims.nbDims = 1;
  kept_min_dims.d[0] = 0;
  kept_max_dims.nbDims = 1;
  kept_max_dims.d[0] = feature_tensor_size;

  ProfileDimension kept_dims = {kept_min_dims, kept_max_dims, kept_max_dims};
  in_profile_dims_["kept"] = kept_dims;

  nvinfer1::Dims rank_min_dims, rank_opt_dims, rank_max_dims;
  rank_min_dims.nbDims = 1;
  rank_min_dims.d[0] = 0;
  rank_opt_dims.nbDims = 1;
  rank_opt_dims.d[0] = feature_tensor_size / 2;
  rank_max_dims.nbDims = 1;
  rank_max_dims.d[0] = feature_tensor_size;

  ProfileDimension rank_dims = {rank_min_dims, rank_opt_dims, rank_max_dims};
  in_profile_dims_["ranks"] = rank_dims;
  in_profile_dims_["indices"] = rank_dims;

  nvinfer1::Dims geom_feats_min_dims = nvinfer1::Dims2{rank_min_dims.d[0], 4};
  nvinfer1::Dims geom_feats_opt_dims = nvinfer1::Dims2{rank_opt_dims.d[0], 4};
  nvinfer1::Dims geom_feats_max_dims = nvinfer1::Dims2{rank_max_dims.d[0], 4};

  ProfileDimension geom_feats_dims = {
    geom_feats_min_dims, geom_feats_opt_dims, geom_feats_max_dims};
  in_profile_dims_["geom_feats"] = geom_feats_dims;
}

NetworkTRT::~NetworkTRT()
{
  context.reset();
  runtime_.reset();
  plan_.reset();
  engine.reset();
}

bool NetworkTRT::init(
  const std::string & onnx_path, const std::string & engine_path, const std::string & precision)
{
  runtime_ =
    tensorrt_common::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    tensorrt_common::LOG_ERROR(logger_) << "Failed to create runtime" << std::endl;
    return false;
  }

  bool success;
  std::ifstream engine_file(engine_path);
  if (engine_file.is_open()) {
    success = loadEngine(engine_path);
  } else {
    auto log_thread = logger_.log_throttle(
      nvinfer1::ILogger::Severity::kINFO,
      "Applying optimizations and building TRT CUDA engine. Please wait a minutes...", 5);
    success = parseONNX(onnx_path, engine_path, precision);
    logger_.stop_throttle(log_thread);
  }

  success &= createContext();

  return success;
}

bool NetworkTRT::setProfile(
  nvinfer1::IBuilder & builder, [[maybe_unused]] nvinfer1::INetworkDefinition & network,
  nvinfer1::IBuilderConfig & config)
{
  auto profile = builder.createOptimizationProfile();

  for (const char * tensor_name : input_tensor_names_) {
    const auto dims = in_profile_dims_[tensor_name];
    profile->setDimensions(tensor_name, nvinfer1::OptProfileSelector::kMIN, dims.min);
    profile->setDimensions(tensor_name, nvinfer1::OptProfileSelector::kOPT, dims.opt);
    profile->setDimensions(tensor_name, nvinfer1::OptProfileSelector::kMAX, dims.max);
  }

  config.addOptimizationProfile(profile);
  return true;
}

bool NetworkTRT::createContext()
{
  if (!engine) {
    tensorrt_common::LOG_ERROR(logger_)
      << "Failed to create context: Engine was not created" << std::endl;
    return false;
  }

  context =
    tensorrt_common::TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
  if (!context) {
    tensorrt_common::LOG_ERROR(logger_) << "Failed to create context" << std::endl;
    return false;
  }

  return true;
}

bool NetworkTRT::parseONNX(
  const std::string & onnx_path, const std::string & engine_path, const std::string & precision,
  const std::size_t workspace_size)
{
  auto builder =
    tensorrt_common::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    tensorrt_common::LOG_ERROR(logger_) << "Failed to create builder" << std::endl;
    return false;
  }

  // TODO(knzo25): make it a parameter
  const std::string plugin_library_path =
    "/workspace/autoware/build/autoware_lidar_bevfusion/libautoware_tensorrt_plugins.so";
  void * const plugin_handle{builder->getPluginRegistry().loadLibrary(plugin_library_path.c_str())};
  if (plugin_handle == nullptr) {
    std::cerr << "Failed to load the plugin library." << std::endl;
    return EXIT_FAILURE;
  }

  auto config =
    tensorrt_common::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    tensorrt_common::LOG_ERROR(logger_) << "Failed to create config" << std::endl;
    return false;
  }
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8400
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
#else
  config->setMaxWorkspaceSize(workspace_size);
#endif
  if (precision == "fp16") {
    if (builder->platformHasFastFp16()) {
      tensorrt_common::LOG_INFO(logger_) << "Using TensorRT FP16 Inference" << std::endl;
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
      tensorrt_common::LOG_INFO(logger_)
        << "TensorRT FP16 Inference isn't supported in this environment" << std::endl;
    }
  }

  std::uint32_t flag{0U};
  // For TensorRT < 10.0, explicit dimension has to be specified to
  // distinguish from the implicit dimension. For TensorRT >= 10.0, explicit
  // dimension is the only choice and this flag has been deprecated.
  if (getInferLibVersion() < 100000) {
    flag |=
      1U << static_cast<std::uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  }

  auto network =
    tensorrt_common::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
  if (!network) {
    tensorrt_common::LOG_ERROR(logger_) << "Failed to create network" << std::endl;
    return false;
  }

  auto parser = tensorrt_common::TrtUniquePtr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, logger_));
  parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));

  std::uint32_t const formats{1U << static_cast<std::uint32_t>(nvinfer1::TensorFormat::kLINEAR)};
  nvinfer1::DataType const dtype{nvinfer1::DataType::kFLOAT};
  network->getInput(0)->setAllowedFormats(formats);
  network->getInput(0)->setType(dtype);
  network->getOutput(0)->setAllowedFormats(formats);
  network->getOutput(0)->setType(dtype);

  if (!setProfile(*builder, *network, *config)) {
    tensorrt_common::LOG_ERROR(logger_) << "Failed to set profile" << std::endl;
    return false;
  }

  plan_ = tensorrt_common::TrtUniquePtr<nvinfer1::IHostMemory>(
    builder->buildSerializedNetwork(*network, *config));
  if (!plan_) {
    tensorrt_common::LOG_ERROR(logger_) << "Failed to create serialized network" << std::endl;
    return false;
  }
  engine = tensorrt_common::TrtUniquePtr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(plan_->data(), plan_->size()));
  if (!engine) {
    tensorrt_common::LOG_ERROR(logger_) << "Failed to create engine" << std::endl;
    return false;
  }

  return saveEngine(engine_path);
}

bool NetworkTRT::saveEngine(const std::string & engine_path)
{
  tensorrt_common::LOG_INFO(logger_) << "Writing to " << engine_path << std::endl;
  std::ofstream file(engine_path, std::ios::out | std::ios::binary);
  file.write(reinterpret_cast<const char *>(plan_->data()), plan_->size());
  return validateNetworkIO();
}

bool NetworkTRT::loadEngine(const std::string & engine_path)
{
  // TODO(knzo25): make it a parameter
  // Load the plugin library.
  const std::string plugin_library_path =
    "/workspace/autoware/build/autoware_lidar_bevfusion/libautoware_tensorrt_plugins.so";
  runtime_->getPluginRegistry().loadLibrary(plugin_library_path.c_str());

  std::ifstream engine_file(engine_path);
  std::stringstream engine_buffer;
  engine_buffer << engine_file.rdbuf();
  std::string engine_str = engine_buffer.str();
  engine = tensorrt_common::TrtUniquePtr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(
    reinterpret_cast<const void *>(engine_str.data()), engine_str.size()));
  tensorrt_common::LOG_INFO(logger_) << "Loaded engine from " << engine_path << std::endl;

  return validateNetworkIO();
}

bool NetworkTRT::validateNetworkIO()
{
  // Whether the number of IO match the expected size
  const auto num_io_tensors = input_tensor_names_.size() + output_tensor_names_.size();
  if (static_cast<std::size_t>(engine->getNbIOTensors()) != num_io_tensors) {
    tensorrt_common::LOG_ERROR(logger_)
      << "Invalid network IO. Expected size: " << num_io_tensors
      << ". Actual size: " << engine->getNbIOTensors() << "." << std::endl;
    throw std::runtime_error("Failed to initialize TRT network.");
  }

  // Log the network IO
  std::string input_tensors = std::accumulate(
    input_tensor_names_.begin(), input_tensor_names_.end(), std::string(),
    [](const std::string & a, const std::string & b) -> std::string { return a + b + " "; });
  std::string output_tensors = std::accumulate(
    output_tensor_names_.begin(), output_tensor_names_.end(), std::string(),
    [](const std::string & a, const std::string & b) -> std::string { return a + b + " "; });
  tensorrt_common::LOG_INFO(logger_) << "Network Inputs: " << input_tensors << std::endl;
  tensorrt_common::LOG_INFO(logger_) << "Network Outputs: " << output_tensors << std::endl;

  // Whether the current engine input profile match the config input profile
  for (const auto tensor_name : input_tensor_names_) {
    ProfileDimension engine_dims{
      engine->getProfileShape(tensor_name, 0, nvinfer1::OptProfileSelector::kMIN),
      engine->getProfileShape(tensor_name, 0, nvinfer1::OptProfileSelector::kOPT),
      engine->getProfileShape(tensor_name, 0, nvinfer1::OptProfileSelector::kMAX)};

    tensorrt_common::LOG_INFO(logger_)
      << "Profile for " << tensor_name << ": " << engine_dims << std::endl;

    if (engine_dims != in_profile_dims_[tensor_name]) {
      tensorrt_common::LOG_ERROR(logger_)
        << "Invalid network input dimension. Config: " << in_profile_dims_[tensor_name]
        << ". Please change the input profile or delete the engine file and build engine again."
        << std::endl;
      throw std::runtime_error("Failed to initialize TRT network.");
    }
  }

  // Whether the IO tensor shapes match the network config, -1 for dynamic size
  validateTensorShape("voxels", {-1, 5});
  validateTensorShape("coors", {-1, 4});
  validateTensorShape(
    "imgs", {-1, 3, static_cast<int>(config_.roi_height_), static_cast<int>(config_.roi_width_)});
  validateTensorShape("lidar2image", {-1, 4, 4});
  validateTensorShape("kept", {-1});
  validateTensorShape("ranks", {-1});
  validateTensorShape("indices", {-1});

  return true;
}

nvinfer1::Dims NetworkTRT::validateTensorShape(
  const char * tensor_name, const std::vector<int> shape)
{
  auto tensor_shape = engine->getTensorShape(tensor_name);
  if (tensor_shape.nbDims != static_cast<int>(shape.size())) {
    tensorrt_common::LOG_ERROR(logger_)
      << "Invalid tensor shape for " << tensor_name << ". Expected size: " << shape.size()
      << ". Actual size: " << tensor_shape.nbDims << "." << std::endl;
    throw std::runtime_error("Failed to initialize TRT network.");
  }
  for (int i = 0; i < tensor_shape.nbDims; ++i) {
    if (tensor_shape.d[i] != static_cast<int>(shape[i])) {
      tensorrt_common::LOG_ERROR(logger_)
        << "Invalid tensor shape for " << tensor_name << ". Expected: " << shape[i]
        << ". Actual: " << tensor_shape.d[i] << "." << std::endl;
      throw std::runtime_error("Failed to initialize TRT network.");
    }
  }
  return tensor_shape;
}

}  // namespace autoware::lidar_bevfusion
