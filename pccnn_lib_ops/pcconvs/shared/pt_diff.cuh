/////////////////////////////////////////////////////////////////////////////
/// \file pt_diff.cuh
///
/// \brief Declaraion of the CUDA operations to compute point differences.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _PT_DIFF_CUH_
#define _PT_DIFF_CUH_

#include "../../shared/defines.cuh"
    
/**
 *  Method to compute the point difference for all neighbors.
 *  @param  pPts            Point tensor.
 *  @param  pFeatures       Point feature tensor.
 *  @param  pSamples        Sample tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pRadii          Radii tensor.
 *  @param  pNormalize      Boolean that indicates if the
 *      point difference is normalized by the radius.
 *  @return     Tensor with the point differences.
 */
torch::Tensor pt_diff(
    torch::Tensor& pPts,
    torch::Tensor& pSamples,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pRadii,
    bool pNormalize);

/**
 *  Method to compute the gradients of the point difference for all neighbors.
 *  @param  pPts            Point tensor.
 *  @param  pFeatures       Point feature tensor.
 *  @param  pSamples        Sample tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pRadii          Radii tensor.
 *  @param  pGradients      Input gradients.
 *  @param  pNormalize      Boolean that indicates if the
 *      point difference is normalized by the radius.
 *  @return     Tensor list with the gradients.
 */
std::vector<torch::Tensor> pt_diff_grads(
    torch::Tensor& pPts,
    torch::Tensor& pSamples,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pRadii,
    torch::Tensor& pGradients,
    bool pNormalize);

#endif