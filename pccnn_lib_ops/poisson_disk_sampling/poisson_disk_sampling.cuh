/////////////////////////////////////////////////////////////////////////////
/// \file pooling_pd.cuh
///
/// \brief Declaraion of the CUDA operations to pool a set of points from
///     a point cloud. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _POISSON_DISK_SAMPLING_CUH_
#define _POISSON_DISK_SAMPLING_CUH_

#include "../shared/defines.cuh"
    
/**
 *  Method to pool a set of points from a point cloud.
 *  @param  pSortedPtsKeys  Sorted point keys tensor.
 *  @param  pNeighbors      Neighbor tensor.
 *  @param  pStartIds       Start ids tensor.
 *  @param  pGridSize       Size of the grid.
 *  @return     Tensor with the identifiers of the 
 *      selected points.
 */
torch::Tensor poisson_disk_sampling(
    torch::Tensor& pSortedPtsKeys,
    torch::Tensor& pNeighbors,
    torch::Tensor& pStartIds,
    torch::Tensor& pGridSize);

#endif