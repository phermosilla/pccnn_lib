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

#include "mcconv.cuh"
#include "./shared/pt_diff.cuh"
#include "./shared/mlp_basis.cuh"
#include "./shared/feat_basis_proj.cuh"
#include "./shared/feat_basis_proj_grads.cuh"
#include "./shared/compute_variance_weights.cuh"
    
///////////////////////// GPU


///////////////////////// CPU

torch::Tensor mcconv(
    torch::Tensor pPts,
    torch::Tensor pPtPDF,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pAxisProj,
    torch::Tensor pAxisBias,
    torch::Tensor pConvWeights)
{
    // Number of mlps.
    int numMLPs = pConvWeights.size(0);

    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // Get the pdf values.
    torch::Tensor neighPDF = torch::index_select(
        pPtPDF, 0, pNeighbors.index({"...", 0}).to(torch::kInt64));

    // MLP basis projection.
    torch::Tensor ptProjs = mlp_basis(ptDiffs, pAxisProj, pAxisBias, 
        neighPDF, true);

    // Compute projected features.
    torch::Tensor projFeat = feat_basis_proj(
        ptProjs, pFeatures, pNeighbors, pStartIds);

    // Perform the matrix multiplication.
    projFeat = torch::reshape(projFeat, {pSamples.size(0), numMLPs, -1});
    projFeat = torch::transpose(projFeat, 0, 1);
    auto outConv = torch::matmul(projFeat, pConvWeights);
    outConv = torch::reshape(torch::transpose(outConv, 0, 1),
        {pSamples.size(0), -1});

    return outConv;
}

std::vector<torch::Tensor> mcconv_grads(
    torch::Tensor pPts,
    torch::Tensor pPtPDF,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pAxisProj,
    torch::Tensor pAxisBias,
    torch::Tensor pConvWeights,
    torch::Tensor pInGradients)
{
    // Number of mlps.
    int numMLPs = pConvWeights.size(0);

    // Reshape the gradients.
    auto inGradient = torch::reshape(pInGradients, 
        {pSamples.size(0), numMLPs, -1});
    inGradient = torch::transpose(inGradient, 0, 1);

    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // Get the pdf values.
    torch::Tensor neighPDF = torch::index_select(
        pPtPDF, 0, pNeighbors.index({"...", 0}).to(torch::kInt64));

    // MLP basis projection.
    torch::Tensor ptProjs = mlp_basis(ptDiffs, pAxisProj, pAxisBias, 
        neighPDF, true);

    // Compute projected features.
    torch::Tensor projFeat = feat_basis_proj(
        ptProjs, pFeatures, pNeighbors, pStartIds);

    // Compute weight gradients.
    projFeat = torch::reshape(projFeat, {pSamples.size(0), numMLPs, -1});
    auto projFeatTranspose = torch::transpose(projFeat, 0, 1);
    projFeatTranspose = torch::transpose(projFeatTranspose, 1, 2);
    auto weightGrads = torch::matmul(projFeatTranspose, inGradient);

    // Compute projected feature gradients.
    auto weightsTranspose = torch::transpose(pConvWeights, 1, 2);
    auto projFeatGrads = torch::matmul(inGradient, weightsTranspose);
    projFeatGrads = torch::transpose(projFeatGrads, 0, 1);
    projFeatGrads = torch::reshape(projFeatGrads, {pSamples.size(0), -1});

    // Compute feature and projection basis gradients (Features, ptBasis).
    auto gradients = feat_basis_proj_grads(ptProjs, pFeatures, 
        pNeighbors, pStartIds, projFeatGrads);

    // Compute gradients basis projection (ptDiff, axis, axisBias, pdf).
    auto gradients2 = mlp_basis_grads(ptDiffs, pAxisProj, pAxisBias, 
        neighPDF, gradients[1], true);

    // Update the pdf gradients.
    auto tensorOptions = torch::TensorOptions().dtype(gradients[1].scalar_type()).
        device(gradients[1].device().type(), gradients[1].device().index());
    auto outPDFGradients = torch::zeros({pPtPDF.size(0)}, tensorOptions);
    auto indexTensor = pNeighbors.index({"...", 0}).to(torch::kInt64);
    outPDFGradients.index_add_(0, indexTensor, gradients2[3]);

    // Compute the gradients of the point clouds.
    auto gradients3 = pt_diff_grads(
        pPts, pSamples, pNeighbors, pRadii, gradients2[0], true);

    return {weightGrads, gradients[0], gradients2[1], gradients2[2], 
        outPDFGradients, gradients3[0], gradients3[1]};
}

torch::Tensor mcconv_compute_weight_variance(
    torch::Tensor pPts,
    torch::Tensor pPtPDF,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pAxisProj,
    torch::Tensor pAxisBias)
{
    // Compute pt differences.
    torch::Tensor ptDiffs = pt_diff(
        pPts, pSamples, pNeighbors, pRadii, true);

    // Get the pdf values.
    torch::Tensor neighPDF = torch::index_select(
        pPtPDF, 0, pNeighbors.index({"...", 0}).to(torch::kInt64));

    // MLP basis projection.
    torch::Tensor ptProjs = mlp_basis(ptDiffs, pAxisProj, pAxisBias, 
        neighPDF, true);

    // Return the variance.
    return compute_variance_weights(
        ptProjs, pFeatures, pNeighbors, pStartIds, false);

}