#include "autoware/mmcv_spconv/common/pytorch_cpp_helper.hpp"
#include "autoware/mmcv_spconv/common/pytorch_device_registry.hpp"

#include <cstdint>

/* 
torch::Tensor IndiceMaxpoolForwardCUDAKernelLauncher(torch::Tensor features,
                                                     torch::Tensor indicePairs,
                                                     torch::Tensor indiceNum,
                                                     std::int64_t numAct); */

/* torch::Tensor indice_maxpool_forward_cuda(torch::Tensor features,
                                          torch::Tensor indicePairs,
                                          torch::Tensor indiceNum,
                                          std::int64_t numAct) {
  return IndiceMaxpoolForwardCUDAKernelLauncher(features, indicePairs,
                                                indiceNum, numAct);
}; */

/* torch::Tensor indice_maxpool_forward_impl(torch::Tensor features,
                                          torch::Tensor indicePairs,
                                          torch::Tensor indiceNum,
                                          std::int64_t numAct);
REGISTER_DEVICE_IMPL(indice_maxpool_forward_impl, CUDA,
                     indice_maxpool_forward_cuda);

torch::Tensor IndiceMaxpoolBackwardCUDAKernelLauncher(torch::Tensor features,
                                                      torch::Tensor outFeatures,
                                                      torch::Tensor outGrad,
                                                      torch::Tensor indicePairs,
                                                      torch::Tensor indiceNum);

torch::Tensor indice_maxpool_backward_cuda(torch::Tensor features,
                                           torch::Tensor outFeatures,
                                           torch::Tensor outGrad,
                                           torch::Tensor indicePairs,
                                           torch::Tensor indiceNum) {
  return IndiceMaxpoolBackwardCUDAKernelLauncher(features, outFeatures, outGrad,
                                                 indicePairs, indiceNum);
};

torch::Tensor indice_maxpool_backward_impl(torch::Tensor features,
                                           torch::Tensor outFeatures,
                                           torch::Tensor outGrad,
                                           torch::Tensor indicePairs,
                                           torch::Tensor indiceNum);

REGISTER_DEVICE_IMPL(indice_maxpool_backward_impl, CUDA,
                     indice_maxpool_backward_cuda) */

torch::Tensor IndiceConvForwardCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor indicePairs,
    torch::Tensor indiceNum, std::int64_t numActOut, std::int64_t _inverse,
    std::int64_t _subM);

torch::Tensor indice_conv_forward_cuda(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       std::int64_t numActOut, std::int64_t _inverse,
                                       std::int64_t _subM) {
  return IndiceConvForwardCUDAKernelLauncher(
      features, filters, indicePairs, indiceNum, numActOut, _inverse, _subM);
};

torch::Tensor indice_conv_forward_impl(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       std::int64_t numActOut, std::int64_t _inverse,
                                       std::int64_t _subM);

REGISTER_DEVICE_IMPL(indice_conv_forward_impl, CUDA, indice_conv_forward_cuda);

/* std::vector<torch::Tensor> IndiceConvBackwardCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, std::int64_t _inverse,
    std::int64_t _subM);

std::vector<torch::Tensor> indice_conv_backward_cuda(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, std::int64_t _inverse,
    std::int64_t _subM) {
  return IndiceConvBackwardCUDAKernelLauncher(
      features, filters, outGrad, indicePairs, indiceNum, _inverse, _subM);
}; */

/* std::vector<torch::Tensor> indice_conv_backward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, std::int64_t _inverse,
    std::int64_t _subM);

REGISTER_DEVICE_IMPL(indice_conv_backward_impl, CUDA,
                     indice_conv_backward_cuda);

torch::Tensor FusedIndiceConvBatchnormCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, std::int64_t numActOut,
    std::int64_t _inverse, std::int64_t _subM);

torch::Tensor fused_indice_conv_batchnorm_forward_cuda(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, std::int64_t numActOut,
    std::int64_t _inverse, std::int64_t _subM) {
  return FusedIndiceConvBatchnormCUDAKernelLauncher(features, filters, bias,
                                                    indicePairs, indiceNum,
                                                    numActOut, _inverse, _subM);
};

torch::Tensor fused_indice_conv_batchnorm_forward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor bias,
    torch::Tensor indicePairs, torch::Tensor indiceNum, std::int64_t numActOut,
    std::int64_t _inverse, std::int64_t _subM);

REGISTER_DEVICE_IMPL(fused_indice_conv_batchnorm_forward_impl, CUDA,
                     fused_indice_conv_batchnorm_forward_cuda)
 */