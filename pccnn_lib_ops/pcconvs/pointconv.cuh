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

#ifndef _POINTCONV_CUH_
#define _POINTCONV_CUH_

#include "../shared/defines.cuh"
    

torch::Tensor pointconv_basis(
    torch::Tensor pPts,
    torch::Tensor& pRadii,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pAxisProj,
    torch::Tensor pAxisBias);

std::vector<torch::Tensor> pointconv_basis_grads(
    torch::Tensor pPts,
    torch::Tensor& pRadii,
    torch::Tensor pSamples,    
    torch::Tensor pNeighbors,
    torch::Tensor pAxisProj,
    torch::Tensor pAxisBias,
    torch::Tensor pGradients);

torch::Tensor pointconv(
    torch::Tensor pPtBasis,
    torch::Tensor pFeatures,    
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pConvWeights);

std::vector<torch::Tensor> pointconv_grads(
    torch::Tensor pPtBasis,
    torch::Tensor pFeatures,   
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds,
    torch::Tensor pConvWeights,
    torch::Tensor pInGradients);

torch::Tensor pointconv_compute_weight_variance(
    torch::Tensor pPtBasis,
    torch::Tensor pFeatures,   
    torch::Tensor pNeighbors,
    torch::Tensor pStartIds);
#endif