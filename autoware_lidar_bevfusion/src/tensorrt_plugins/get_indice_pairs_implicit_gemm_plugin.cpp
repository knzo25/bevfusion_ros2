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

#include "autoware/tensorrt_plugins/get_indice_pairs_implicit_gemm_plugin.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>

#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>

#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

// GetIndicePairsImplicitGemm

namespace nvinfer1
{
namespace plugin
{

GetIndicePairsImplicitGemmPlugin::GetIndicePairsImplicitGemmPlugin(
  const std::string & name, GetIndicePairsImplicitGemmParameters const & params)
: layer_name_{name}, params_{params}
{
  initFieldsToSerialize();
}

void GetIndicePairsImplicitGemmPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  data_to_serialize_.emplace_back("batch_size", &params_.batch_size, PluginFieldType::kINT32, 1);
  data_to_serialize_.emplace_back("algo", &params_.algo, PluginFieldType::kINT32, 1);
  data_to_serialize_.emplace_back("is_train", &params_.is_train, PluginFieldType::kINT32, 1);
  data_to_serialize_.emplace_back(
    "dilation_dims", &params_.dilation_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back("ksize_dims", &params_.ksize_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back(
    "out_padding_dims", &params_.out_padding_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back("padding_dims", &params_.padding_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back(
    "spatial_shape_dims", &params_.spatial_shape_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back("stride_dims", &params_.stride_dims, PluginFieldType::kDIMS, 1);
  data_to_serialize_.emplace_back("subm", &params_.subm, PluginFieldType::kINT32, 1);
  data_to_serialize_.emplace_back("transpose", &params_.transpose, PluginFieldType::kINT32, 1);

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * GetIndicePairsImplicitGemmPlugin::getCapabilityInterface(
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

IPluginV3 * GetIndicePairsImplicitGemmPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new GetIndicePairsImplicitGemmPlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * GetIndicePairsImplicitGemmPlugin::getPluginName() const noexcept
{
  return kGET_INDICE_PAIRS_IMPLICIT_GEMM_PLUGIN_NAME;
}

char const * GetIndicePairsImplicitGemmPlugin::getPluginVersion() const noexcept
{
  return kGET_INDICE_PAIRS_IMPLICIT_GEMM_PLUGIN_VERSION;
}

char const * GetIndicePairsImplicitGemmPlugin::getPluginNamespace() const noexcept
{
  return kGET_INDICE_PAIRS_IMPLICIT_GEMM_PLUGIN_NAMESPACE;
}

std::int32_t GetIndicePairsImplicitGemmPlugin::getNbOutputs() const noexcept
{
  return 5;
}

std::int32_t GetIndicePairsImplicitGemmPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 5);
  PLUGIN_ASSERT(in[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(out[1].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(out[2].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(out[3].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[4].desc.dims.nbDims == 0);

  std::int64_t kernel_volume = 1;
  for (const std::int64_t ksize : params_.ksize) {
    kernel_volume *= ksize;
  }
  
  PLUGIN_ASSERT(in[0].desc.dims.d[1] == 4);  // coords + 1
  
  PLUGIN_ASSERT(out[0].desc.dims.d[1] == 4);  // coords + 1
  PLUGIN_ASSERT(out[0].desc.type == in[0].desc.type);

  PLUGIN_ASSERT(out[1].desc.dims.d[0] == kernel_volume);
  PLUGIN_ASSERT(out[1].desc.type == in[0].desc.type);

  PLUGIN_ASSERT(out[2].desc.dims.d[1] == 1);
  PLUGIN_ASSERT(out[2].desc.type == in[0].desc.type);

  PLUGIN_ASSERT(out[3].desc.type == in[0].desc.type);

  PLUGIN_ASSERT(out[4].desc.type == in[0].desc.type);

  return 0;
}

bool GetIndicePairsImplicitGemmPlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 5);

  return (
    in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
    in_out[pos].desc.type == nvinfer1::DataType::kINT32);
}

std::int32_t GetIndicePairsImplicitGemmPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 5);

  output_types[0] = input_types[0];
  output_types[1] = input_types[0];
  output_types[2] = input_types[0];
  output_types[3] = input_types[0];
  output_types[4] = input_types[0];

  return 0;
}

std::int32_t GetIndicePairsImplicitGemmPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs, IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  PLUGIN_ASSERT(num_outputs == 5);
  PLUGIN_ASSERT(inputs[0].nbDims == 2);

  std::int64_t kernel_volume = 1;

  for (std::size_t i = 0; i < params_.ksize.size(); ++i) {
    kernel_volume *= params_.ksize[i];
  }

  if (params_.subm) {
    outputs[0] = inputs[0];
    outputs[0].d[1] = inputs[0].d[1];

    outputs[1].nbDims = 2;
    outputs[1].d[0] = expr_builder.constant(kernel_volume);
    outputs[1].d[1] = inputs[0].d[0];

    outputs[2].nbDims = 2;
    outputs[2].d[0] = inputs[0].d[0];
    outputs[2].d[1] = expr_builder.constant(1);

    outputs[3].nbDims = 1;
    outputs[3].d[0] = inputs[0].d[0];

  } else {

    auto opt_value = expr_builder.operation(
      DimensionOperation::kCEIL_DIV, *inputs[0].d[0], *expr_builder.constant(2));

    outputs[0].nbDims = 2;
    outputs[0].d[0] = expr_builder.declareSizeTensor(4, *opt_value, *expr_builder.constant(out_inds_num_limit_));
    outputs[0].d[1] = inputs[0].d[1];

    outputs[1].nbDims = 2;
    outputs[1].d[0] = expr_builder.constant(kernel_volume);
    outputs[1].d[1] = expr_builder.declareSizeTensor(4, *opt_value, *expr_builder.constant(out_inds_num_limit_));

    outputs[2].nbDims = 2;
    outputs[2].d[0] = expr_builder.declareSizeTensor(4, *opt_value, *expr_builder.constant(out_inds_num_limit_));
    outputs[2].d[1] = expr_builder.constant(1);

    outputs[3].nbDims = 1;
    outputs[3].d[0] = expr_builder.declareSizeTensor(4, *opt_value, *expr_builder.constant(out_inds_num_limit_));
    
  }

  /**
   * out_inds.shape, out_inds.dtype
(torch.Size([66693, 4]), torch.int32)
pair_fwd.shape, pair_fwd.dtype
(torch.Size([27, 66693]), torch.int32)
pair_mask_fwd_splits.shape, pair_mask_fwd_splits.dtype
(torch.Size([66693, 1]), torch.int32)
mask_argsort_fwd_splits.shape, mask_argsort_fwd_splits.dtype
(torch.Size([66693]), torch.int32)

  */


  // num_activate_out
  outputs[4].nbDims = 0;

  return 0;
}

std::int32_t GetIndicePairsImplicitGemmPlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  using SpconvOps = spconvlib::spconv::csrc::sparse::all::SpconvOps;
  using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;

  bool is_subm = params_.subm;
  // direct table: a hash based algorithm that don't need unique. enabled
  // by default.
  bool direct_table = true;
  // only regular conv need direct table.
  bool use_direct_table = direct_table && !is_subm;
  int static_num_act_in = out_inds_num_limit_;

  std::vector<int> ksize(params_.ksize.begin(), params_.ksize.end());
  std::vector<int> stride(params_.stride.begin(), params_.stride.end());
  std::vector<int> padding(params_.padding.begin(), params_.padding.end());
  std::vector<int> dilation(params_.dilation.begin(), params_.dilation.end());
  std::vector<int> input_dims(params_.spatial_shape.begin(), params_.spatial_shape.end());


  auto out_dims = SpconvOps::get_conv_output_size(input_dims, ksize, stride,
                                                  padding, dilation);
  std::vector<int64_t> output_dims_i64(out_dims.begin(), out_dims.end());
  // if shape is too large, we will use slower int64->int32 hash table instead
  // of int32->int32 table.
  int64_t out_spatial_volume =
      std::accumulate(output_dims_i64.begin(), output_dims_i64.end(),
                      int64_t(1), std::multiplies<int64_t>());

  bool use_int64_hash_k =
      out_spatial_volume >= int64_t(std::numeric_limits<int>::max());



  /* std::transform(params_.ksize.begin(), params_.ksize.end(), ksize.begin(), 
                  [](const auto & x) { return static_cast<int>(x); });
  std::transform(params_.stride.begin(), params_.stride.end(), stride.begin(), 
                  [](const auto & x) { return static_cast<int>(x); });
  std::transform(params_.padding.begin(), params_.padding.end(), padding.begin(), 
                  [](const auto & x) { return static_cast<int>(x); });
  std::transform(params_.dilation.begin(), params_.dilation.end(), dilation.begin(), 
                  [](const auto & x) { return static_cast<int>(x); });
  std::transform(params_.spatial_shape.begin(), params_.spatial_shape.end(), input_dims.begin(), 
                  [](const auto & x) { return static_cast<int>(x); }); */

  int kernel_volume = 1;
  for (const auto & ksize : params_.ksize) {
    kernel_volume *= ksize;
  }

  auto max_act_out_theory = SpconvOps::get_handcrafted_max_act_out(
      input_desc[0].dims.d[0], ksize, stride, padding, dilation);
  // query workspace size.
  /* int workspace_size = SpconvOps::get_indice_gen_workspace_size(
      kernel_volume, out_inds_num_limit_, out_inds_num_limit_, out_inds_num_limit_,
      is_subm, use_int64_hash_k, use_direct_table); */
  // you should return workspace size in tensorrt plugin method.
  //tv::Tensor workspace = tv::empty({workspace_size}, tv::uint8, 0);
  /* tv::Tensor workspace_tensor =
      tv::from_blob(workspace, {workspace_size}, tv::uint8, 0); */
  // get tensor map required by pair gen from workspace
  // keep in mind that our indice gen function use a "allocator" to alloc
  // temp/out tensors, in python we use TorchAllocator which is a simple
  // dynamic allocator, in c++ (inference engine) we need to use
  // fixed-size workspace and create a static allocator.
  auto ws_tensors = SpconvOps::get_indice_gen_tensors_from_workspace(
      reinterpret_cast<std::uint8_t*>(workspace), kernel_volume, out_inds_num_limit_,
      is_subm ? out_inds_num_limit_ : out_inds_num_limit_,
      max_act_out_theory, is_subm, use_int64_hash_k, use_direct_table);
  // pair can also have a upper bound.
  // !!!!!IMPORTANT!!!!!!! if you provide a static (padded) pair_fwd and
  // other indice data, the output layout is tight pair_fwd_correct =
  // pair_fwd_padded.view(-1)[:KV * real_pair_size].view(KV,
  // real_pair_size) this valid for pair_fwd, pair_bwd, pair_mask_fwd,
  // pair_mask_bwd, mask_argsort_fwd, mask_argsort_bwd.
  int pair_fwd_size_padded =
      is_subm ? input_desc[0].dims.d[0] : out_inds_num_limit_;
  tv::Tensor pair_fwd_padded =
      tv::from_blob(outputs[1], {kernel_volume, pair_fwd_size_padded}, tv::int32, 0);
  // you can find equivalent python code of following code in python
  // package
  bool is_split_mask =
      params_.algo == static_cast<std::int64_t>(tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm);
  int mask_count = is_split_mask ? 2 : 1;
  
  /* tv::Tensor pair_mask_fwd_padded =
      tv::empty({mask_count, pair_fwd_size_padded}, tv::int32, 0);
  tv::Tensor mask_argsort_fwd_padded =
      tv::empty({mask_count, pair_fwd_size_padded}, tv::int32, 0);
  tv::Tensor out_inds = tv::empty(
      {is_subm ? input_desc[0].desc.dims.d[0] : out_inds_num_limit_, 4},
      tv::int32, 0);
  tv::Tensor indices_kernel_num = tv::zeros({kernel_volume}, tv::int32, 0); */

  tv::Tensor pair_mask_fwd_padded =
      tv::from_blob(outputs[2], {mask_count, pair_fwd_size_padded}, tv::int32, 0);
  tv::Tensor mask_argsort_fwd_padded =
      tv::from_blob(outputs[3], {mask_count, pair_fwd_size_padded}, tv::int32, 0);
  tv::Tensor out_inds = tv::from_blob(outputs[0], 
      {is_subm ? input_desc[0].dims.d[0] : out_inds_num_limit_, 4},
      tv::int32, 0);
  tv::Tensor indices_kernel_num = tv::zeros({kernel_volume}, tv::int32, 0);
  
  tv::Tensor input_indices =
      tv::from_blob(inputs[0], {input_desc[0].dims.d[0], 4}, tv::int32, 0);

  std::tuple<tv::Tensor, int> pair_res;

  tv::Context ctx;
  ctx.set_cuda_stream_int(reinterpret_cast<std::uintptr_t>(stream));
  
  if (is_subm) {
    // subm out inds equal to input inds, just copy them
    out_inds.copy_(input_indices, ctx);

    // subm exmaple
    // create output tensors and insert them to static allocator
    // output tensors needed in subm get_indice_pairs_implicit_gemm,
    // saved to static allocator.
    ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair_fwd_padded});
    ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK, pair_mask_fwd_padded});
    ws_tensors.insert(
        {SPCONV_ALLOC_MASK_ARG_SORT, mask_argsort_fwd_padded});
    ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds});
    ws_tensors.insert(
        {SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num});
    StaticAllocator alloc(ws_tensors);

    pair_res = SpconvOps::get_indice_pairs_implicit_gemm(
        alloc, input_indices, params_.batch_size, input_dims,
        static_cast<int>(params_.algo), ksize, stride, padding, dilation,
        {0, 0, 0}, params_.subm, params_.transpose, false /*is_train*/,
        reinterpret_cast<std::uintptr_t>(stream), out_inds_num_limit_,
        tv::CUDAKernelTimer(false), use_direct_table);
    // for subm num_act_out always equal to num_act_in_real

  } else {
    // WARNING be careful with inverse conv, understand python
    // code first. no inverse example here.
    // regular conv need more outputs, used for inversed conv.
    // bwd shape is [KV, static num_act_in (previous num_act_out_bound)]
    tv::Tensor pair_bwd_padded =
        tv::empty({kernel_volume, static_num_act_in}, tv::int32, 0);
    tv::Tensor pair_mask_bwd_padded =
        tv::empty({mask_count, static_num_act_in}, tv::int32, 0);
    tv::Tensor mask_argsort_bwd_padded =
        tv::empty({mask_count, static_num_act_in}, tv::int32, 0);

    ws_tensors.insert({SPCONV_ALLOC_PAIR_FWD, pair_fwd_padded});
    ws_tensors.insert({SPCONV_ALLOC_PAIR_BWD, pair_bwd_padded});

    ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK, pair_mask_fwd_padded});
    ws_tensors.insert({SPCONV_ALLOC_PAIR_MASK_BWD, pair_mask_bwd_padded});

    ws_tensors.insert(
        {SPCONV_ALLOC_MASK_ARG_SORT, mask_argsort_fwd_padded});
    ws_tensors.insert(
        {SPCONV_ALLOC_MASK_ARG_SORT_BWD, mask_argsort_bwd_padded});

    ws_tensors.insert({SPCONV_ALLOC_OUT_INDICES, out_inds});
    ws_tensors.insert(
        {SPCONV_ALLOC_INDICE_NUM_PER_LOC, indices_kernel_num});

    StaticAllocator alloc(ws_tensors);

    pair_res = SpconvOps::get_indice_pairs_implicit_gemm(
        alloc, input_indices, params_.batch_size, input_dims,
        static_cast<int>(params_.algo), ksize, stride, padding, dilation,
        {0, 0, 0}, params_.subm, params_.transpose, false /*is_train*/,
        reinterpret_cast<std::uintptr_t>(stream), out_inds_num_limit_,
        tv::CUDAKernelTimer(false), use_direct_table);
  }
  // after get pair datas, we can start to do real convolution!
  // in static inference engine, you need to split pair-gen and conv to
  // different layers to reuse pair data
  // here we just use previous result.
  std::int32_t num_act_out_real = std::get<1>(pair_res);
  std::int32_t * num_act_out_data = static_cast<std::int32_t *>(outputs[4]);

  cudaError_t const status = cudaMemcpyAsync(
    num_act_out_data, &num_act_out_real, sizeof(std::int32_t), cudaMemcpyHostToDevice, stream);

  return status;
}

std::int32_t GetIndicePairsImplicitGemmPlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * GetIndicePairsImplicitGemmPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * GetIndicePairsImplicitGemmPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t GetIndicePairsImplicitGemmPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  using SpconvOps = spconvlib::spconv::csrc::sparse::all::SpconvOps;

  bool is_subm = params_.subm;
  // direct table: a hash based algorithm that don't need unique. enabled
  // by default.
  bool direct_table = true;
  // only regular conv need direct table.
  bool use_direct_table = direct_table && !is_subm;
  
  
  std::vector<int> ksize(params_.ksize.begin(), params_.ksize.end());
  std::vector<int> stride(params_.stride.begin(), params_.stride.end());
  std::vector<int> padding(params_.padding.begin(), params_.padding.end());
  std::vector<int> dilation(params_.dilation.begin(), params_.dilation.end());
  std::vector<int> input_dims(params_.spatial_shape.begin(), params_.spatial_shape.end());


  auto out_dims = SpconvOps::get_conv_output_size(input_dims, ksize, stride,
                                                  padding, dilation);
  std::vector<int64_t> output_dims_i64(out_dims.begin(), out_dims.end());
  // if shape is too large, we will use slower int64->int32 hash table instead
  // of int32->int32 table.
  int64_t out_spatial_volume =
      std::accumulate(output_dims_i64.begin(), output_dims_i64.end(),
                      int64_t(1), std::multiplies<int64_t>());
  bool use_int64_hash_k =
      out_spatial_volume >= int64_t(std::numeric_limits<int>::max());

  int kernel_volume = 1;
  for (const auto & ksize : params_.ksize) {
    kernel_volume *= ksize;
  }

  // query workspace size.
  int workspace_size = SpconvOps::get_indice_gen_workspace_size(
      kernel_volume, out_inds_num_limit_, out_inds_num_limit_, out_inds_num_limit_,
      is_subm, use_int64_hash_k, use_direct_table);

  return static_cast<std::size_t>(workspace_size);
}

}  // namespace plugin
}  // namespace nvinfer1
