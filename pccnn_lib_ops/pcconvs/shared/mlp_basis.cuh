/////////////////////////////////////////////////////////////////////////////
/// \file mlp_basis.cuh
///
/// \brief Declaraion of the CUDA operations to compute the projection of
///     the points into an mlp basis.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _MLP_BASIS_CUH_
#define _MLP_BASIS_CUH_

#include "../../shared/defines.cuh"
    
/**
 *  Method to compute the point projection into an mlp basis.
 *  @param  pPts            Point tensor.
 *  @param  pAxisProj       Axes in which project points.
 *  @param  pBiasAxis       Bias used to project the points.
 *  @param  pPDFs           PDF of each point.
 *  @param  pDiv            Boolean that indicates if the pdf 
 *      weight divide or multiply the basis function.
 *  @return     Tensor with the point projection.
 */
torch::Tensor mlp_basis(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pAxisProj,    
    torch::Tensor& pBiasAxis,
    torch::Tensor& pPDFs,
    const bool pDiv);


/**
 *  Method to compute the gradients of thepoint projection into an mlp basis.
 *  @param  pPts            Point tensor.
 *  @param  pAxisProj       Axes in which project points.
 *  @param  pBiasAxis       Bias used to project the points.
 *  @param  pPDFs           PDF of each point.
 *  @param  pInGradients    Input gradients.
 *  @param  pDiv            Boolean that indicates if the pdf 
 *      weight divide or multiply the basis function.
 *  @return     List of tensors with the gradients.
 */
std::vector<torch::Tensor> mlp_basis_grads(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pAxisProj,    
    torch::Tensor& pBiasAxis,
    torch::Tensor& pPDFs,
    torch::Tensor& pInGradients,
    const bool pDiv);

#endif