/////////////////////////////////////////////////////////////////////////////
/// \file kpconvn.cuh
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

#include "kpconvn.cuh"
#include "./shared/pt_diff.cuh"
#include "./shared/kernel_pts_basis.cuh"
#include "./shared/feat_basis_proj.cuh"
#include "./shared/feat_basis_proj_grads.cuh"
#include "./shared/compute_variance_weights.cuh"
    
///////////////////////// GPU


///////////////////////// CPU

torch::Tensor kpconvn(
    torch::Tensor pPts,
    torch::Tensor pPtPDF,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pKPoints,
    float pSigma,
    torch::Tensor pConvWeights)
{
    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);


    // Basis projection.
    torch::Tensor ptProjs = kernel_pts_basis(
        ptDiffs, pKPoints, pSigma, pNeighbors);

    // Get the pdf values.
    torch::Tensor neighPDF = torch::index_select(
        pPtPDF, 0, pNeighbors.index({"...", 0}).to(torch::kInt64));
    ptProjs = ptProjs / torch::reshape(neighPDF, {ptProjs.size(0), 1});

    // Compute projected features.
    torch::Tensor projFeat = feat_basis_proj(
        ptProjs, pFeatures, pNeighbors, pStartIds);

    // Perform the matrix multiplication.
    projFeat = torch::reshape(projFeat, {pSamples.size(0), -1});
    auto outConv = torch::matmul(projFeat, pConvWeights);

    return outConv;
}


std::vector<torch::Tensor> kpconvn_grads(
    torch::Tensor pPts,
    torch::Tensor pPtPDF,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pKPoints,
    float pSigma,
    torch::Tensor pConvWeights,
    torch::Tensor pInGradients)
{
    auto inGradient = pInGradients;

    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // MLP basis projection.
    torch::Tensor ptProjs = kernel_pts_basis(
        ptDiffs, pKPoints, pSigma, pNeighbors);

    // Get the pdf values.
    torch::Tensor neighPDF = torch::index_select(
        pPtPDF, 0, pNeighbors.index({"...", 0}).to(torch::kInt64));
    ptProjs = ptProjs / torch::reshape(neighPDF, {ptProjs.size(0), 1});

    // Compute projected features.
    torch::Tensor projFeat = feat_basis_proj(
        ptProjs, pFeatures, pNeighbors, pStartIds);

    // Compute weight gradients.
    projFeat = torch::reshape(projFeat, {pSamples.size(0), -1});
    auto projFeatTranspose = torch::transpose(projFeat, 0, 1);
    auto weightGrads = torch::matmul(projFeatTranspose, inGradient);

    // Compute projected feature gradients.
    auto weightsTranspose = torch::transpose(pConvWeights, 0, 1);
    auto projFeatGrads = torch::matmul(inGradient, weightsTranspose);

    // Compute feature and projection basis gradients (Features, ptBasis).
    auto gradients = feat_basis_proj_grads(ptProjs, pFeatures, 
        pNeighbors, pStartIds, projFeatGrads);

    // Compute gradients basis projection.
    auto auxGradient = gradients[1] / torch::reshape(neighPDF, {ptProjs.size(0), 1});
    auto gradients2 = kernel_pts_basis_grads(
        ptDiffs, pKPoints, pSigma, pNeighbors, auxGradient);

    // Compute the gradients of the point clouds.
    auto gradients3 = pt_diff_grads(
        pPts, pSamples, pNeighbors, pRadii, gradients2[0], true);

    return {weightGrads, gradients[0], gradients2[1], 
        gradients3[0], gradients3[1]};
}


torch::Tensor kpconvn_compute_weight_variance(
    torch::Tensor pPts,
    torch::Tensor pPtPDF,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pKPoints,
    float pSigma)
{
    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // MLP basis projection.
    torch::Tensor ptProjs = kernel_pts_basis(
        ptDiffs, pKPoints, pSigma, pNeighbors);

    // Get the pdf values.
    torch::Tensor neighPDF = torch::index_select(
        pPtPDF, 0, pNeighbors.index({"...", 0}).to(torch::kInt64));
    ptProjs = ptProjs / torch::reshape(neighPDF, {ptProjs.size(0), 1});

    // Return the variance.
    return compute_variance_weights(
        ptProjs, pFeatures, pNeighbors, pStartIds, false);
}