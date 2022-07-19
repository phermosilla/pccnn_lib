/////////////////////////////////////////////////////////////////////////////
/// \file compute_keys.cu
///
/// \brief Implementation of the CUDA operations to compute the keys indices 
///     of a point cloud into a regular grid. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "../shared/defines.cuh"
#include "../shared/math_helper.cuh"
#include "../shared/cuda_kernel_utils.cuh"
#include "../shared/grid_utils.cuh"
#include "../shared/gpu_device_utils.cuh"

#include "compute_keys.cuh"

///////////////////////// GPU

/**
 *  GPU kernel to compute the keys of each point.
 *  @param  pNumPts         Number of points.
 *  @param  pPts            Array of points.
 *  @param  pBatchIds       Array of batch ids.
 *  @param  pAABBMin       Array of minimum point of bounding
 *      boxes.
 *  @param  pNumCells       Number of cells.
 *  @param  pInvCellSize    Inverse cell size.
 *  @param  pOutKeys        Output array with the point keys.
 */
template<int D>
__global__ void compute_keys_gpu_kernel(
    const unsigned int pNumPts,
    const float* __restrict__ pPts,
    const int* __restrict__ pBatchIds,
    const float* __restrict__ pAABBMin,
    const int* __restrict__ pNumCells,
    const float* __restrict__ pInvCellSize,
    mccnn::int64_m* __restrict__ pOutKeys)
{
    // Create the pointers.
    const mccnn::fpoint<D>* ptsPtr = (const mccnn::fpoint<D>*)pPts;
    const int* batchIdsPtr = pBatchIds;
    const mccnn::fpoint<D>* aabbMinPtr = (const mccnn::fpoint<D>*)pAABBMin;
    const mccnn::ipoint<D> numCellsPtr(pNumCells);
    const mccnn::fpoint<D> invCellSizePtr(pInvCellSize);

    // Create the indices.
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    for(int curPtIndex = initPtIndex; curPtIndex < pNumPts; curPtIndex += totalThreads)
    {
        //Get the values for the point.
        int curBatchId = batchIdsPtr[curPtIndex];
        mccnn::fpoint<D> curPt = ptsPtr[curPtIndex];
        mccnn::fpoint<D> curSAABBMin = aabbMinPtr[curBatchId];

        //Compute the current cell indices.
        mccnn::ipoint<D> cell = mccnn::compute_cell_gpu_funct(
            curPt, curSAABBMin, numCellsPtr, invCellSizePtr);

        //Compute the key index of the cell.
        mccnn::int64_m keyIndex = mccnn::compute_key_gpu_funct(
            cell, numCellsPtr, curBatchId);

        //Save the key index.
        pOutKeys[curPtIndex] = keyIndex;
    }
}

///////////////////////// CPU 

torch::Tensor compute_keys(
    torch::Tensor pPts,
    torch::Tensor pBatchIds,
    torch::Tensor pAABBMin,
    torch::Tensor pGridSize,
    torch::Tensor pCellSize) {

    // Get the number of dimensions and points.
    int numDims = pPts.size(1);
    int numPts = pPts.size(0);

    // Get device properties.
    cudaDeviceProp props = mccnn::get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    DIMENSION_FUNCT_PTR(numDims, compute_keys_gpu_kernel, funcPtr);

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;
    unsigned int numBlocks = mccnn::get_max_active_block_x_sm(
        blockSize, funcPtr, 0);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = numPts/blockSize;
    execBlocks += (numPts%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Create output.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kInt64).
        device(pPts.device().type(), pPts.device().index());
    auto outTensor = torch::zeros({numPts}, tensorOptions);

    // Compute inverse of the cell size.
    auto invCellSize = torch::reciprocal(pCellSize);
    auto invCellSize2 = torch::reshape(invCellSize, {1, -1});

    // Call the cuda kernel.
    DIMENSION_SWITCH_CALL(numDims, compute_keys_gpu_kernel, totalNumBlocks, blockSize, 0,
        numPts, 
        (const float*)pPts.data_ptr(), 
        (const int*)pBatchIds.data_ptr(), 
        (const float*)pAABBMin.data_ptr(), 
        (const int*)pGridSize.data_ptr(), 
        (const float*)invCellSize.data_ptr(), 
        (mccnn::int64_m*)outTensor.data_ptr());

    return outTensor;
}
