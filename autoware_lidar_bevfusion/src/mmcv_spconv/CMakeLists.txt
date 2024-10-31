cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mmcv_spconv)

#list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/python3.10/dist-packages/torch/share/cmake/")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions("-DMMCV_WITH_CUDA")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_NO_HALF_OPERATORS__") #  -G -g --generate-line-info 

set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(mmcv_spconv SHARED
    pytorch/cuda/spconv_ops_cuda.cu
    pytorch/cuda/sparse_reordering.cu
    pytorch/cuda/sparse_indice.cu
    pytorch/cuda/cudabind.cpp
    pytorch/cpu/sparse_indice.cpp
    pytorch/cpu/sparse_reordering.cpp
    pytorch/spconv_ops.cpp    
)

include_directories(
    pytorch
    common
    common/cuda
)

target_link_libraries(mmcv_spconv "${TORCH_LIBRARIES}")
set_property(TARGET mmcv_spconv PROPERTY CXX_STANDARD 17)

install(TARGETS mmcv_spconv EXPORT mmcv_spconvConfig DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)