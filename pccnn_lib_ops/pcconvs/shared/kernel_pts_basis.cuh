/////////////////////////////////////////////////////////////////////////////
/// \file kernel_pts_basis.cuh
///
/// \brief Declaraion of the CUDA operations to compute the points 
///     correlation to a set of kernel points
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _KPOINTS_BASIS_CUH_
#define _KPOINTS_BASIS_CUH_

#include "../../shared/defines.cuh"
    
/**
 *  Method to compute the point correlation to a set of kernel points.
 *  @param  pPts            Point tensor.
 *  @param  pKernelPts      Axes in which project points.
 *  @param  pSigma          Bias used to project the points.
 *  @param  pNeighbors      Neighbors.
 *  @return     Tensor with the point correlations.
 */
torch::Tensor kernel_pts_basis(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pKernelPts,    
    float pSigma,
    torch::Tensor& pNeighbors);
torch::Tensor kernel_pts_basis_gauss(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pKernelPts,    
    float pSigma,
    torch::Tensor& pNeighbors);


/**
 *  Method to compute the gradients of point correlation to a set of kernel points.
 *  @param  pPts            Point tensor.
 *  @param  pKernelPts      Axes in which project points.
 *  @param  pSigma          Bias used to project the points.
 *  @param  pNeighbors      Neighbors.
 *  @param  pInGradients    Input gradients.
 *  @return     List of tensors with the gradients.
 */
std::vector<torch::Tensor> kernel_pts_basis_grads(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pKernelPts,    
    float pSigma,
    torch::Tensor& pNeighbors,
    torch::Tensor& pInGradients);
std::vector<torch::Tensor> kernel_pts_basis_gauss_grads(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pKernelPts,    
    float pSigma,
    torch::Tensor& pNeighbors,
    torch::Tensor& pInGradients);



#endif