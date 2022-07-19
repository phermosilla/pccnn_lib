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

#include "../shared/defines.cuh"
#include "../shared/math_helper.cuh"
#include "../shared/cuda_kernel_utils.cuh"
#include "../shared/grid_utils.cuh"
#include "../shared/gpu_device_utils.cuh"

#include "pointconv.cuh"
#include "./shared/pt_diff.cuh"
#include "./shared/feat_basis_proj.cuh"
#include "./shared/feat_basis_proj_grads.cuh"
#include "./shared/compute_variance_weights.cuh"
    
///////////////////////// GPU


///////////////////////// CPU

torch::Tensor pointconv_basis(
    torch::Tensor pPts,
    torch::Tensor& pRadii,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pAxisProj,
    torch::Tensor pAxisBias)
{
    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // Project into the axes.
    torch::Tensor pointProj = torch::matmul(ptDiffs, pAxisProj);

    // Add the bias.
    pointProj = pointProj + pAxisBias;

    return pointProj;
}

std::vector<torch::Tensor> pointconv_basis_grads(
    torch::Tensor pPts,
    torch::Tensor& pRadii,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pAxisProj,
    torch::Tensor pAxisBias,
    torch::Tensor pGradients)
{
    // Gradient bias.
    auto outAxisBiasGradients = torch::sum(pGradients, 0, true);
    
    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // Gradient axis.
    auto ptDiffsTranspose = torch::transpose(ptDiffs, 0, 1);
    auto outAxisGradients = torch::matmul(ptDiffsTranspose, pGradients);

    // Gradient pt diffs.
    auto axisProjTranspose = torch::transpose(pAxisProj, 0, 1);
    auto outPtDiffGradients = torch::matmul(pGradients, axisProjTranspose);

    // Compute the gradients of the point clouds.
    auto auxGradients = pt_diff_grads(
        pPts, pSamples, pNeighbors, pRadii, outPtDiffGradients, true);

    return {auxGradients[0], auxGradients[1], outAxisGradients, outAxisBiasGradients};
}

torch::Tensor pointconv(
    torch::Tensor pPtBasis,
    torch::Tensor pFeatures,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pConvWeights)
{
    // Compute projected features.
    torch::Tensor projFeat = feat_basis_proj(
        pPtBasis, pFeatures, pNeighbors, pStartIds);

    // Perform the matrix multiplication.
    projFeat = torch::reshape(projFeat, {pStartIds.size(0), -1});
    auto outConv = torch::matmul(projFeat, pConvWeights);

    return outConv;
}

std::vector<torch::Tensor> pointconv_grads(
    torch::Tensor pPtBasis,
    torch::Tensor pFeatures,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pConvWeights,
    torch::Tensor pInGradients)
{
    auto inGradient = pInGradients;

    // Compute projected features.
    torch::Tensor projFeat = feat_basis_proj(
        pPtBasis, pFeatures, pNeighbors, pStartIds);

    // Compute weight gradients.
    projFeat = torch::reshape(projFeat, {pStartIds.size(0), -1});
    auto projFeatTranspose = torch::transpose(projFeat, 0, 1);
    auto weightGrads = torch::matmul(projFeatTranspose, inGradient);

    // Compute projected feature gradients.
    auto weightsTranspose = torch::transpose(pConvWeights, 0, 1);
    auto projFeatGrads = torch::matmul(inGradient, weightsTranspose);

    // Compute feature and projection basis gradients (Features, ptBasis).
    auto gradients = feat_basis_proj_grads(pPtBasis, pFeatures, 
        pNeighbors, pStartIds, projFeatGrads);

    return {weightGrads, gradients[0], gradients[1]};
}

torch::Tensor pointconv_compute_weight_variance(
    torch::Tensor pPtBasis,
    torch::Tensor pFeatures,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds)
{
    // Return the variance.
    return compute_variance_weights(
        pPtBasis, pFeatures, pNeighbors, pStartIds, false);
}