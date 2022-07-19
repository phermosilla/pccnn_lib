/////////////////////////////////////////////////////////////////////////////
/// \file build_grid_ds.cuh
///
/// \brief Declaration of the CUDA operations to build the data structure to 
///     access the sparse regular grid. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _BUILD_GRID_DS_H_
#define _BUILD_GRID_DS_H_

#include "../shared/defines.cuh"

/**
 *  Operation to build a grid data structure.
 *  @param  pKeys           Tensor with the point keys.
 *  @param  pGridSize       Tensor with the size of the grid.
 *  @param  pOutShape       Vector with the size of the output tensor.
 */
torch::Tensor build_grid_ds(
    torch::Tensor pKeys,
    torch::Tensor pGridSize,
    std::vector<int64_t>& pOutShape);

#endif