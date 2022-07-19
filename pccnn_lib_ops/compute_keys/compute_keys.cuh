/////////////////////////////////////////////////////////////////////////////
/// \file compute_keys.cuh
///
/// \brief Declaration of the CUDA operations to compute the keys indices 
///     of a point cloud into a regular grid.  
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _COMPUTE_KEYS_H_
#define _COMPUTE_KEYS_H_

#include "../shared/defines.cuh"

/**
 *  Operation to compute the keys of a point cloud.
 *  @param  pPts           Tensor with the point coordinates.
 *  @param  pBatchIds      Tensor with the batch ids.
 *  @param  pAABBMin       Tensor with the aabb min coordainte.
 *  @param  pGridSize      Tensor with the grid size.
 *  @param  pCellSize      Tensor with the cell size.
 */
 torch::Tensor compute_keys(
    torch::Tensor pPts,
    torch::Tensor pBatchIds,
    torch::Tensor pAABBMin,
    torch::Tensor pGridSize,
    torch::Tensor pCellSize);

#endif