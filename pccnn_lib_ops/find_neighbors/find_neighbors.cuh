/////////////////////////////////////////////////////////////////////////////
/// \file find_neighbors.cu
///
/// \brief Implementation of the CUDA operations to find the neighboring
///     points.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _FIND_NEIGHBORS_H_
#define _FIND_NEIGHBORS_H_

#include "../shared/defines.cuh"

/**
 *  Method to find the neighbors of points in the grid data structure.
 *  @param  pPts            Tensor with the points.
 *  @param  pPtsKeys        Tensor with the point keys.
 *  @param  pSample         Tensor with the samples.
 *  @param  pSampleKeys     Tensor with the sample keys.
 *  @param  pGridDS         Tensor with the grid data structure.
 *  @param  pGridSize       Tensor with the grid size.
 *  @param  pRadii          Tensor with the radius.
 *  @param  pMaxNeighbors   Maximum number of neighbors. If 0, all
 *      neighbors are considered.
 *  @returns    Tensor with the neighbor indices and Tensor with the
 *      end indices per sample to the neighbor list.
 */
std::vector<torch::Tensor> find_neighbors(
    torch::Tensor pPts,
    torch::Tensor pPtsKeys,
    torch::Tensor pSample,
    torch::Tensor pSampleKeys,
    torch::Tensor pGridDS,
    torch::Tensor pGridSize,
    torch::Tensor pRadii,
    unsigned int pMaxNeighbors);

#endif