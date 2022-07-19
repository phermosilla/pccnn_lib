/////////////////////////////////////////////////////////////////////////////
/// \file feat_basis_proj.cuh
///
/// \brief Declaraion of the CUDA operations to compute the projection of
///     the features into the basis.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _FEAT_BASIS_PROJ_CUH_
#define _FEAT_BASIS_PROJ_CUH_

#include "../../shared/defines.cuh"
    
/**
 *  Method to compute the point projection into an mlp basis.
 *  @param  pPtBasis        Point basis tensor.
 *  @param  pPtFeatures     Point feature tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pStartIds       Start ids tensor.
 *  @return     Tensor with the feature projection.
 */
torch::Tensor feat_basis_proj(
    torch::Tensor& pPtBasis,
    torch::Tensor& pPtFeatures,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pStartIds);

#endif