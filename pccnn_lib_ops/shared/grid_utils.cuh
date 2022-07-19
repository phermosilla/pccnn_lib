/////////////////////////////////////////////////////////////////////////////
/// \file grid_utils.cuh
///
/// \brief Utilities for the cuda implementations of the tensor operations.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef GRID_UTILS_H_
#define GRID_UTILS_H_

#include <stdio.h>

#include "defines.cuh"
#include "math_helper.cuh"

namespace mccnn{

    ///////////////////////// DEVICE FUNCTIONS

    /**
     *  Function to compute the total number of cells.
     *  @param  pPosition       Position of the point.
     *  @param  pSMinPoint      Minimum point of the bounding box scaled by the
     *      inverse of the cell size.
     *  @param  pNumCells       Number of cells.
     *  @param  pInvCellSize    Cell size.
     *  @return Cell indices.
     *  @paramT D               Number of dimensions.
     */
    template<int D>
    __device__ __forceinline__ mccnn::int64_m 
    compute_total_num_cells_gpu_funct(
        const mccnn::ipoint<D> pNumCells)
    {
        mccnn::int64_m result = 1;
#pragma unroll
        for(int i = 0; i < D; ++i)
            result *= pNumCells[i];
        return result;
    }

    /**
     *  Function to compute the cell for a given point.
     *  @param  pPosition       Position of the point.
     *  @param  pSMinPoint      Minimum point of the bounding box scaled by the
     *      inverse of the cell size.
     *  @param  pNumCells       Number of cells.
     *  @param  pInvCellSize    Cell size.
     *  @return Cell indices.
     *  @paramT D               Number of dimensions.
     */
    template<int D>
    __device__ __forceinline__ mccnn::ipoint<D> 
    compute_cell_gpu_funct(
        const mccnn::fpoint<D> pPosition,
        const mccnn::fpoint<D> pSMinPoint,
        const mccnn::ipoint<D> pNumCells,
        const mccnn::fpoint<D> pInvCellSize)
    {
        mccnn::fpoint<D> relPoint = (pPosition - pSMinPoint)*pInvCellSize;
        mccnn::ipoint<D> curCell = (mccnn::ipoint<D>)floorp(relPoint); 
        return minp(maxp(curCell, mccnn::ipoint<D>(0)), pNumCells-1);
    }


    /**
     *  Function to compute the key for a given cell.
     *  @param  pCell       Cell index.
     *  @param  pNumCells   Number of cells.
     *  @param  pBatchId    Current batch id.
     *  @return Key of the cell.
     *  @paramT D           Number of dimensions.
     */
    template<int D>
    __device__ __forceinline__ mccnn::int64_m compute_key_gpu_funct(
        const mccnn::ipoint<D> pCell,
        const mccnn::ipoint<D> pNumCells,
        const int pBatchId)
    {
        mccnn::int64_m key = 0;
        mccnn::int64_m accumKey = 1;
#pragma unroll
        for(int i = D-1; i >=0 ; --i)
        {
            key += pCell[i]*accumKey;
            accumKey *= pNumCells[i];
        }
        return key + accumKey*pBatchId;
    }

    /**
     *  Function to compute the cell from a given key.
     *  @param  pKey        Input key.
     *  @param  pNumCells   Number of cells.
     *  @return Cell indices and batch id (b, d1, d2, ..., dn).
     *  @paramT D           Number of dimensions.
     */
     template<int D>
    __device__ __forceinline__ mccnn::ipoint<D+1> compute_cell_from_key_gpu_funct(
        const mccnn::int64_m pKey,
        const mccnn::ipoint<D> pNumCells)
    {
        mccnn::int64_m auxInt = pKey;
        mccnn::ipoint<D+1> result;
#pragma unroll
        for(int i = D-1; i >=0; --i){
            result[i+1] = auxInt%pNumCells[i];
            auxInt = auxInt/pNumCells[i];
        }
        result[0] = auxInt;

        return result;
    }

    /**
     *  Function to compute data structure index from a given key.
     *  @param  pKey        Input key.
     *  @param  pNumCells   Number of cells.
     *  @return Index to the data structure.
     *  @paramT D           Number of dimensions.
     */
     template<int D>
    __device__ __forceinline__ int compute_ds_index_from_key_gpu_funct(
        const mccnn::int64_m pKey,
        const mccnn::ipoint<D> pNumCells)
    {
        mccnn::int64_m divVal = 1;
#pragma unroll
        for(int i = 0; i < D-2; ++i)
            divVal *= pNumCells[i+2];
        return (int)(pKey/divVal);
    }

    /**
     *  Function to compute data structure index from a given cell.
     *  @param  pExtCell    Input cell in which the batch id is in the
     *      first position.
     *  @param  pNumCells   Number of cells.
     *  @return Index to the data structure.
     *  @paramT D           Number of dimensions.
     */
     template<int D>
    __device__ __forceinline__ int compute_ds_index_from_cell_gpu_funct(
        const mccnn::ipoint<D+1> pExtCell,
        const mccnn::ipoint<D> pNumCells)
    {
        return pExtCell[0]*pNumCells[0]*pNumCells[1]+
                pExtCell[1]*pNumCells[1] + pExtCell[2];
    }
}

#endif