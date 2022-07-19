/////////////////////////////////////////////////////////////////////////////
/// \file pccnn.cuh
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

#include "pccnn.cuh"
#include "./shared/pt_diff.cuh"
#include "./shared/kernel_pts_basis.cuh"
#include "./shared/feat_basis_proj.cuh"
#include "./shared/feat_basis_proj_grads.cuh"
#include "./shared/compute_variance_weights.cuh"
    
///////////////////////// GPU


///////////////////////// CPU

torch::Tensor pccnn(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pPtPDF,
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

    // MLP basis projection.
    torch::Tensor ptProjs = kernel_pts_basis_gauss(
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


std::vector<torch::Tensor> pccnn_grads(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pPtPDF,
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
    torch::Tensor ptProjs = kernel_pts_basis_gauss(
        ptDiffs, pKPoints, pSigma, pNeighbors);

    // Get the pdf values.
     auto indexTensor = pNeighbors.index({"...", 0}).to(torch::kInt64);
    torch::Tensor neighPDF = torch::reshape(torch::index_select(
        pPtPDF, 0, indexTensor), {ptProjs.size(0), 1});
    ptProjs = ptProjs / neighPDF;

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

    // Compute the pdf gradients.
    auto outPDFGradients = (-1.0*ptProjs*gradients[1])/neighPDF;
    outPDFGradients = torch::sum(outPDFGradients, 1);
    auto tensorOptions = torch::TensorOptions().dtype(outPDFGradients.scalar_type()).
        device(outPDFGradients.device().type(), outPDFGradients.device().index());
    auto scatteredOutPDFGradients = torch::zeros({pPtPDF.size(0)}, tensorOptions);
    scatteredOutPDFGradients.index_add_(0, indexTensor, outPDFGradients);

    // Compute gradients basis projection.
    auto inKPtsGrads = gradients[1]/neighPDF;
    auto gradients2 = kernel_pts_basis_gauss_grads(
        ptDiffs, pKPoints, pSigma, pNeighbors, inKPtsGrads);

    // Compute the gradients of the point clouds.
    auto gradients3 = pt_diff_grads(
        pPts, pSamples, pNeighbors, pRadii, gradients2[0], true);

    // Weight, features, kernel_pts, pdfs, pts, samples
    return {weightGrads, gradients[0], gradients2[1], scatteredOutPDFGradients,
        gradients3[0], gradients3[1]};
}


torch::Tensor pccnn_compute_weight_variance(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pPtPDF,
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
    torch::Tensor ptProjs = kernel_pts_basis_gauss(
        ptDiffs, pKPoints, pSigma, pNeighbors);

    // Get the pdf values.
    torch::Tensor neighPDF = torch::index_select(
        pPtPDF, 0, pNeighbors.index({"...", 0}).to(torch::kInt64));
    ptProjs = ptProjs / torch::reshape(neighPDF, {ptProjs.size(0), 1});

    // Return the variance.
    return compute_variance_weights(
        ptProjs, pFeatures, pNeighbors, pStartIds, false);
}
