/////////////////////////////////////////////////////////////////////////////
/// \file feat_basis_proj.cuh
///
/// \brief Declaraion of the CUDA operations to compute the gradients of
///     projection of the features into the basis.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _FEAT_BASIS_PROJ_GRADS_CUH_
#define _FEAT_BASIS_PROJ_GRADS_CUH_

#include "../../shared/defines.cuh"
    
/**
 *  Method to compute the gradients of the 
 *  point projection into an mlp basis.
 *  @param  pPtBasis        Point basis tensor.
 *  @param  pPtFeatures     Point feature tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pStartIds       Start ids tensor.
 *  @param  pInGradients    Input gradients.
 *  @return     Tensor list with the gradients.
 */
std::vector<torch::Tensor> feat_basis_proj_grads(
    torch::Tensor& pPtBasis,
    torch::Tensor& pPtFeatures,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pStartIds,
    torch::Tensor& pInGradients);

#endif