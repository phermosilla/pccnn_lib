/////////////////////////////////////////////////////////////////////////////
/// \file pcconv.cuh
///
/// \brief Implementation of the CUDA operations to compute a point
///     convolution. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "../../shared/defines.cuh"
#include "../../shared/math_helper.cuh"
#include "../../shared/cuda_kernel_utils.cuh"
#include "../../shared/grid_utils.cuh"
#include "../../shared/gpu_device_utils.cuh"

#include "pt_diff.cuh"
    
///////////////////////// GPU


///////////////////////// CPU

torch::Tensor pt_diff(
    torch::Tensor& pPts,
    torch::Tensor& pSamples,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pRadii,
    bool pNormalize)
{
    // Get the point coordinates.
    torch::Tensor neighPts = torch::index_select(
        pPts, 0, pNeighbors.index({"...", 0}).to(torch::kInt64));

    // Get the sample coordiantes.
    torch::Tensor neighSamples = torch::index_select(
        pSamples, 0, pNeighbors.index({"...", 1}).to(torch::kInt64));

    // Compute the difference.
    if(pNormalize){
        torch::Tensor invRadii = torch::reciprocal(pRadii);
        return (neighPts - neighSamples)*invRadii;
    }else{
        return (neighPts - neighSamples);
    }
}

std::vector<torch::Tensor> pt_diff_grads(
    torch::Tensor& pPts,
    torch::Tensor& pSamples,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pRadii,
    torch::Tensor& pInGradients,
    bool pNormalize)
{
    // Compute the gradients.
    auto curGrads = pInGradients;
    if(pNormalize){
        torch::Tensor invRadii = torch::reciprocal(pRadii);
        curGrads = curGrads*invRadii;
    }

    auto tensorOptions = torch::TensorOptions().dtype(pInGradients.scalar_type()).
        device(pInGradients.device().type(), pInGradients.device().index());
    auto outPtsGradients = torch::zeros({pPts.size(0), pPts.size(1)}, tensorOptions);
    auto outSamplesGradients = torch::zeros({pSamples.size(0), pSamples.size(1)}, tensorOptions);

    auto indexTensorPts = pNeighbors.index({"...", 0}).to(torch::kInt64);
    auto indexTensorSamples = pNeighbors.index({"...", 1}).to(torch::kInt64);
    
    outPtsGradients.index_add_(0, indexTensorPts, curGrads);
    outSamplesGradients.index_add_(0, indexTensorSamples, curGrads);

    outSamplesGradients = outSamplesGradients * -1.0;

    return {outPtsGradients, outSamplesGradients};
}