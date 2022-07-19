/////////////////////////////////////////////////////////////////////////////
/// \file sphconv.cuh
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

#include "../shared/defines.cuh"
#include "../shared/math_helper.cuh"
#include "../shared/cuda_kernel_utils.cuh"
#include "../shared/grid_utils.cuh"
#include "../shared/gpu_device_utils.cuh"

#include "mcconv.cuh"
#include "./shared/pt_diff.cuh"
#include "./shared/spherical_basis.cuh"
#include "./shared/feat_basis_proj.cuh"
#include "./shared/feat_basis_proj_grads.cuh"
#include "./shared/compute_variance_weights.cuh"
    
///////////////////////// GPU


///////////////////////// CPU

torch::Tensor sphconv(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pRadialBins,
    torch::Tensor pAzimuzBins,    
    torch::Tensor pPolarBins,
    torch::Tensor pConvWeights)
{
    // Number of mlps.
    int numMLPs = pConvWeights.size(0);

    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // Spherical basis projection.
    torch::Tensor ptProjs = spherical_basis(
        ptDiffs, pRadialBins, pAzimuzBins, pPolarBins);

    // Compute projected features.
    torch::Tensor projFeat = feat_basis_proj(
        ptProjs, pFeatures, pNeighbors, pStartIds);

    // Perform the matrix multiplication.
    projFeat = torch::reshape(projFeat, {pStartIds.size(0), -1});
    auto outConv = torch::matmul(projFeat, pConvWeights);

    // Get number of points.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).
        device(pStartIds.device().type(), pStartIds.device().index());
    auto numPtsTensor = torch::zeros({pStartIds.size(0)}, tensorOptions);
    auto onesTensor = torch::ones_like(pNeighbors.index({"...", 0})).to(torch::kFloat32);
    numPtsTensor.index_add_(0, pNeighbors.index({"...", 1}).to(torch::kInt64), onesTensor);

    return outConv / (torch::reshape(numPtsTensor, {-1, 1})+1e-6);
}

std::vector<torch::Tensor> sphconv_grads(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pRadialBins,
    torch::Tensor pAzimuzBins,    
    torch::Tensor pPolarBins,
    torch::Tensor pConvWeights,
    torch::Tensor pInGradients)
{
    // Get number of points.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).
        device(pStartIds.device().type(), pStartIds.device().index());
    auto numPtsTensor = torch::zeros({pStartIds.size(0)}, tensorOptions);
    auto onesTensor = torch::ones_like(pNeighbors.index({"...", 0})).to(torch::kFloat32);
    numPtsTensor.index_add_(0, pNeighbors.index({"...", 1}).to(torch::kInt64), onesTensor);

    // Reshape the gradients.
    auto inGradient = pInGradients / (torch::reshape(numPtsTensor, {-1, 1})+1e-6);

    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // Spherical basis projection.
    torch::Tensor ptProjs = spherical_basis(
        ptDiffs, pRadialBins, pAzimuzBins, pPolarBins);

    // Compute projected features.
    torch::Tensor projFeat = feat_basis_proj(
        ptProjs, pFeatures, pNeighbors, pStartIds);

    // Compute weight gradients.
    projFeat = torch::reshape(projFeat, {pStartIds.size(0), -1});
    auto projFeatTranspose = torch::transpose(projFeat, 0, 1);
    auto weightGrads = torch::matmul(projFeatTranspose, inGradient);

    // Compute projected feature gradients.
    auto weightsTranspose = torch::transpose(pConvWeights, 0, 1);
    auto projFeatGrads = torch::matmul(inGradient, weightsTranspose);

    // Compute feature and projection basis gradients (Features, ptBasis).
    auto gradients = feat_basis_proj_grads(ptProjs, pFeatures, 
        pNeighbors, pStartIds, projFeatGrads);

    return {weightGrads, gradients[0]};
}

torch::Tensor sphconv_compute_weight_variance(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pRadialBins,
    torch::Tensor pAzimuzBins,    
    torch::Tensor pPolarBins)
{
    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // MLP basis projection.
    torch::Tensor ptProjs = spherical_basis(
        ptDiffs, pRadialBins, pAzimuzBins, pPolarBins);

    // Return the variance.
    return compute_variance_weights(
        ptProjs, pFeatures, pNeighbors, pStartIds, true);

}