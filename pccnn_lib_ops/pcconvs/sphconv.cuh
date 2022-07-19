/////////////////////////////////////////////////////////////////////////////
/// \file sphconv.cuh
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

#ifndef _SPHCONV_CUH_
#define _SPHCONV_CUH_

#include "../shared/defines.cuh"
    
torch::Tensor sphconv(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pRadialBins,
    torch::Tensor pAzimuzBins,    
    torch::Tensor pPolarBins,
    torch::Tensor pConvWeights);


std::vector<torch::Tensor> sphconv_grads(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pRadialBins,
    torch::Tensor pAzimuzBins,    
    torch::Tensor pPolarBins,
    torch::Tensor pConvWeights,
    torch::Tensor pInGradients);


torch::Tensor sphconv_compute_weight_variance(
    torch::Tensor pPts,
    torch::Tensor pFeatures,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pRadii,
    torch::Tensor pRadialBins,
    torch::Tensor pAzimuzBins,    
    torch::Tensor pPolarBins);
#endif