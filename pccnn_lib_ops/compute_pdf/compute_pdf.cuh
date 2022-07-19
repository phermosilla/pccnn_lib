/////////////////////////////////////////////////////////////////////////////
/// \file compute_pdf.cuh
///
/// \brief Declaraion of the CUDA operations to compute the pdf of each 
///     neighboring point. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _COMPUTE_PDF_CUH_
#define _COMPUTE_PDF_CUH_

#include "../shared/defines.cuh"

/**
 *  Method to compute the pdfs on the gpu.
 *  @param  pManifoldDims   Dimensions of the data manifold.
 *  @param  pBandWidth      Bandwidth for pdf computation.
 *  @param  pPts            Input point tensor.
 *  @param  pNeighbors      Input neighbors tensor.
 *  @param  pStartI         Input start indices tensor.
 *  @return     Tensor with the resulting pdf.   
 */
torch::Tensor compute_pdf(
    const int pManifoldDims,
    const torch::Tensor& pBandWidth,
    const torch::Tensor& pPts,
    const torch::Tensor& pNeighbors,
    const torch::Tensor& pStartI);

/**
 *  Method to compute the gradients of the points wrt the pdfs values.
 *  @param  pManifoldDims   Dimensions of the data manifold.
 *  @param  pBandWidth      Bandwidth for pdf computation.
 *  @param  pPts            Input point tensor.
 *  @param  pNeighbors      Input neighbors tensor.
 *  @param  pStartI         Input start indices tensor.
 *  @param  pInGrads        Input gradients.
 *  @return     Tensor with the gradients of the points.             
 */
torch::Tensor compute_pdf_grads(
    const int pManifoldDims,
    const torch::Tensor& pBandWidth,
    const torch::Tensor& pPts,
    const torch::Tensor& pNeighbors,
    const torch::Tensor& pStartI,
    const torch::Tensor& pInGrads);

#endif