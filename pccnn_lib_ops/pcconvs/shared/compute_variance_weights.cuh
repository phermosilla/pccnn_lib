/////////////////////////////////////////////////////////////////////////////
/// \file compute_variance_weights.cuh
///
/// \brief Declaraion of the operations to compute the variance of the 
///     convolution weights.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _COMPUTE_VARIANCE_WEIGHTS_CUH_
#define _COMPUTE_VARIANCE_WEIGHTS_CUH_

#include "../../shared/defines.cuh"
    
/**
 *  Method to compute the variance of the convolution weights..
 *  @param  pPtBasis            Point basis tensor.
 *  @param  pPtFeatures         Point feature tensor.
 *  @param  pNeighbors          Neighbor tensor.
 *  @param  pStartIds           Start ids tensor.
 *  @param  pDivideNumNeighs    Boolean that indicates if
 *      we will divide the convolution result by the number of neighbors.
 *  @return     Tensor with the variance.
 */
torch::Tensor compute_variance_weights(
    torch::Tensor& pPtBasis,
    torch::Tensor& pPtFeatures,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pStartIds,
    bool pDivideNumNeighs);

#endif