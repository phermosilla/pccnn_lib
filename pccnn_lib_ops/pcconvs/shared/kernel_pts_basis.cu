/////////////////////////////////////////////////////////////////////////////
/// \file kernel_pts_basis.cu
///
/// \brief Implementation of the CUDA operations to compute the points 
///     correlation to a set of kernel points
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "../../shared/defines.cuh"

#include "kernel_pts_basis.cuh"
    
///////////////////////// GPU


///////////////////////// CPU

torch::Tensor kernel_pts_basis(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pKernelPts,    
    float pSigma,
    torch::Tensor& pNeighbors)
{
    int numDims = pPtDiffs.size(1);

    // Project into the axes.
    torch::Tensor diffKPPoints = torch::reshape(pKernelPts, {1, -1, numDims}) -
        torch::reshape(pPtDiffs, {-1, 1, numDims});
    auto distances = torch::mul(diffKPPoints, diffKPPoints);
    distances = torch::sum(distances, -1);
    distances = torch::sqrt(torch::clamp(distances, 0.0));
    
    return torch::clamp(1.0-(distances / pSigma), 0.0, 1.0);
}

std::vector<torch::Tensor> kernel_pts_basis_grads(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pKernelPts,    
    float pSigma,
    torch::Tensor& pNeighbors,
    torch::Tensor& pInGradients)
{
    int numDims = pPtDiffs.size(1);
    int numKernelPts = pKernelPts.size(0);

    // Project into the axes.
    torch::Tensor diffKPPoints = torch::reshape(pKernelPts, {1, -1, numDims}) -
        torch::reshape(pPtDiffs, {-1, 1, numDims});
    auto distances = torch::mul(diffKPPoints, diffKPPoints);
    distances = torch::sum(distances, -1);
    distances = torch::sqrt(torch::clamp(distances, 0.0));
    auto result = 1.0-(distances / pSigma);
    distances = torch::reciprocal(torch::reshape(distances+1e-6, {-1, numKernelPts, 1}));

    auto mask = (result >= 0.0).to(torch::kFloat32);
    auto curGradient = pInGradients * mask;

    curGradient = torch::reshape(- curGradient / pSigma, {-1, numKernelPts, 1});
    curGradient = curGradient * distances * diffKPPoints;
    
    auto gradPts = -torch::sum(curGradient, 1);
    auto gradKernelPts = torch::sum(curGradient, 0);

    return {gradPts, gradKernelPts};
}

torch::Tensor kernel_pts_basis_gauss(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pKernelPts,    
    float pSigma,
    torch::Tensor& pNeighbors)
{
    int numDims = pPtDiffs.size(1);
    
    torch::Tensor diffKPPoints = torch::reshape(pKernelPts, {1, -1, numDims}) -
        torch::reshape(pPtDiffs, {-1, 1, numDims});
    auto distances = torch::mul(diffKPPoints, diffKPPoints);
    distances = torch::sum(distances, -1);
    
    distances = torch::exp(-(distances / (2.0f*pSigma*pSigma)));
    return distances;
}

std::vector<torch::Tensor> kernel_pts_basis_gauss_grads(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pKernelPts,    
    float pSigma,
    torch::Tensor& pNeighbors,
    torch::Tensor& pInGradients)
{
    int numDims = pPtDiffs.size(1);
    int numKernelPts = pKernelPts.size(0);

    // Project into the axes.
    torch::Tensor diffKPPoints = torch::reshape(pKernelPts, {1, -1, numDims}) -
        torch::reshape(pPtDiffs, {-1, 1, numDims});
    auto distances = torch::mul(diffKPPoints, diffKPPoints);
    distances = torch::sum(distances, -1, true);
    distances = torch::exp(-(distances / (2.0f*pSigma*pSigma)));
    
    auto grad_in = torch::reshape(pInGradients, {-1, numKernelPts, 1});
    auto gradKernelPts = -grad_in*distances*diffKPPoints*(1.f/(pSigma*pSigma));
    auto gradPts = torch::sum(-gradKernelPts, 1);
    gradKernelPts = torch::sum(gradKernelPts, 0);
    return {gradPts, gradKernelPts};
}
