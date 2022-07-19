/////////////////////////////////////////////////////////////////////////////
/// \file mcconv.cuh
///
/// \brief Declaraion of the CUDA operations to compute a point
///     convolution. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _MCCONV_CUH_
#define _MCCONV_CUH_

#include "../shared/defines.cuh"
    
/**
 *  Method to compute a point convolution.
 *  @param  pPts            Point tensor.
 *  @param  pPtPDF          Point pdf.
 *  @param  pFeatures       Point feature tensor.
 *  @param  pSamples        Sample tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pStartIds       Start ids tensor.
 *  @param  pRadii          Radii tensor.
 *  @param  pAxisProj       Projection axis parameters.
 *  @param  pConvWeights    Convolution weights.
 *  @return     Tensor with the convoluted features.
 */
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
    torch::Tensor pConvWeights);

/**
 *  Method to compute the gradients of a point convolution.
 *  @param  pPts            Point tensor.
 *  @param  pPtPDF          Point pdf.
 *  @param  pFeatures       Point feature tensor.
 *  @param  pSamples        Sample tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pStartIds       Start ids tensor.
 *  @param  pRadii          Radii tensor.
 *  @param  pAxisProj       Projection axis parameters.
 *  @param  pConvWeights    Convolution weights.
 *  @param  pInGradients    Input gradients.
 *  @return     Vector of ouput tensor with the gradients.
 */
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
    torch::Tensor pInGradients);

/**
 *  Method to compute the variance of the weights.
 *  @param  pPts            Point tensor.
 *  @param  pPtPDF          Point pdf.
 *  @param  pFeatures       Point feature tensor.
 *  @param  pSamples        Sample tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pStartIds       Start ids tensor.
 *  @param  pRadii          Radii tensor.
 *  @param  pAxisProj       Projection axis parameters.
 *  @return     Tensor with the convoluted features.
 */
torch::Tensor mcconv_compute_weight_variance(
    torch::Tensor pPts,
    torch::Tensor pPtPDF,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pAxisProj,
    torch::Tensor pAxisBias);
#endif