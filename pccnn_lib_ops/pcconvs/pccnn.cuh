/////////////////////////////////////////////////////////////////////////////
/// \file pccnn.cuh
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

#ifndef _PCCNN_CUH_
#define _PCCNN_CUH_

#include "../shared/defines.cuh"
    
/**
 *  Method to compute a point convolution.
 *  @param  pPts            Point tensor.
 *  @param  pFeatures       Point feature tensor.
 *  @param  pSamples        Sample tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pStartIds       Start ids tensor.
 *  @param  pRadii          Radii tensor.
 *  @param  pKPoints        Kernel points.
 *  @param  pSigma          Sigma used for the correlation.
 *  @param  pConvWeights    Convolution weights.
 *  @return     Tensor with the convoluted features.
 */
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
    torch::Tensor pConvWeights);

/**
 *  Method to compute the gradients of a point convolution.
 *  @param  pPts            Point tensor.
 *  @param  pFeatures       Point feature tensor.
 *  @param  pSamples        Sample tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pStartIds       Start ids tensor.
 *  @param  pRadii          Radii tensor.
 *  @param  pKPoints        Kernel points.
 *  @param  pSigma          Sigma used for the correlation.
 *  @param  pConvWeights    Convolution weights.
 *  @param  pInGradients    Input gradients.
 *  @return     Vector of ouput tensor with the gradients.
 */
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
 *  @param  pKPoints        Kernel points.
 *  @param  pSigma          Sigma used for the correlation.
 *  @return     Tensor with the convoluted features.
 */
torch::Tensor pccnn_compute_weight_variance(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pPtPDF,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pKPoints,
    float pSigma);
#endif