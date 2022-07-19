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

#include "../shared/defines.cuh"
#include "../shared/math_helper.cuh"
#include "../shared/cuda_kernel_utils.cuh"
#include "../shared/grid_utils.cuh"
#include "../shared/gpu_device_utils.cuh"

#include "poisson_disk_sampling.cuh"
    
///////////////////////// GPU

/**
 *  GPU kernel to count the unique keys in a list.
 *  @param  pNumPts             Number of points.
 *  @param  pNumUniqueKeys      Number of unique keys.
 *  @param  pCurrentCellBlock   Selected cells to process.
 *  @param  pNumCells           Number of cells.
 *  @param  pKeys               Input cell keys.
 *  @param  pNeighbors          Input point neighbors.
 *  @param  pNeighStartIds      Input start ids.
 *  @param  pUniqueKeyStartIds  Input key start ids.
 *  @param  pUsed               Input/Output used vector.
 */
 template <int D>
 __global__ void select_points_pd_gpu_kernel(
    const unsigned int pNumPts,
    const unsigned int pNumUniqueKeys,
    const int* pCurrentCellBlock,
    const int* __restrict__ pNumCells,
    const mccnn::int64_m* pKeys,
    const int2* __restrict__ pNeighbors,
    const int* __restrict__ pNeighStartIds,
    const int* __restrict__ pUniqueKeyStartIds,
    int* __restrict__ pUsed)
{
    // Get pointers.
    const mccnn::ipoint<D> curCellBlock(pCurrentCellBlock);
    const mccnn::ipoint<D> numCells(pNumCells);

    // Get the global thread index.
    int initKeyIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curKeyIndex = initKeyIndex; 
        curKeyIndex < pNumUniqueKeys; 
        curKeyIndex += totalThreads)
    {
        
        // Get the key and batch id.
        mccnn::int64_m curKey = pKeys[curKeyIndex];
        mccnn::ipoint<D+1> cellIndex = mccnn::compute_cell_from_key_gpu_funct(
            curKey, numCells);

        // Check that it is a valid cell.
        bool validCell = true;
#pragma unroll
        for(int dimIter = 0; dimIter < D; ++dimIter)
            validCell = validCell && ((cellIndex[dimIter+1]%2)==curCellBlock[dimIter]);

        if(validCell){

            // Find the start index.
            int startPtIndex = (curKeyIndex > 0)? 
                pUniqueKeyStartIds[curKeyIndex-1]: 0;
            int endPtIndex = pUniqueKeyStartIds[curKeyIndex];
            
            // Iterate until there are no more points in the cell.
            for(int curPtIndex = startPtIndex; 
                curPtIndex < endPtIndex;
                ++curPtIndex)
            {
                // Check the range of neighbors.
                int startNeighIndex = (curPtIndex > 0)? 
                    pNeighStartIds[curPtIndex-1]: 0;
                int endNeighIndex = pNeighStartIds[curPtIndex];

                // Check if some of the neighbors have been previously selected.
                bool valid = true;
                for(int neighIter = startNeighIndex; 
                    neighIter < endNeighIndex;
                    ++neighIter)
                {
                    int neighIndex = pNeighbors[neighIter].x;
                    valid = valid && (pUsed[neighIndex] == 0);
                }
                
                // If this is a valid point, we select it.
                if(valid){
                    pUsed[curPtIndex] = 1;
                }
            }
        }
    }
}

///////////////////////// CPU

torch::Tensor poisson_disk_sampling(
    torch::Tensor& pSortedPtsKeys,
    torch::Tensor& pNeighbors,
    torch::Tensor& pStartIds,
    torch::Tensor& pGridSize)
{
    // Compute the unique cell keys
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> uniqueKeysTuple =
        torch::unique_consecutive(pSortedPtsKeys, true, true);
    torch::Tensor accumNumPts = torch::cumsum(
        (torch::Tensor)std::get<2>(uniqueKeysTuple),
        0, torch::kInt32);

    // Get variables.
    int numKeys = accumNumPts.size(0);
    int numPts = pSortedPtsKeys.size(0);
    int numDims = pGridSize.size(0);

    // Get device properties.
    cudaDeviceProp props = mccnn::get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    DIMENSION_FUNCT_PTR(numDims, select_points_pd_gpu_kernel, funcPtr);

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;
    unsigned int numBlocks = mccnn::get_max_active_block_x_sm(
        blockSize, funcPtr, 0);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = numKeys/blockSize;
    execBlocks += (numKeys%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Create the output tensor.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kInt32)
        .device(pSortedPtsKeys.device().type(), 
        pSortedPtsKeys.device().index());
    torch::Device cudaDevice(pSortedPtsKeys.device().type(), 
        pSortedPtsKeys.device().index());
    auto selected = torch::zeros({numPts}, tensorOptions);

    //Compute the number of calls we need to do.
    unsigned int numCalls = 1;
    for(int i = 0; i < numDims; ++i)
        numCalls *= 2;

    //Iterate over the different calls required.
    for(int callIter = 0; callIter < numCalls; ++callIter)
    {

        //Compute valid cells mod.
        int auxInt = callIter;
        int* validCode = new int[numDims];
        for(int i = 0; i < numDims; ++i){
            validCode[i] = auxInt%2;
            auxInt = auxInt/2;
        }

        // Upload valid cells to gpu.
        auto tensorOptions = torch::TensorOptions().dtype(torch::kInt32);
        torch::Device cudaDevice(pSortedPtsKeys.device().type(), 
            pSortedPtsKeys.device().index());
        auto validCellTensor = torch::from_blob(&validCode[0], 
            {numDims}, tensorOptions).to(cudaDevice);

        delete[] validCode;

        // Call the cuda kernel.
        DIMENSION_SWITCH_CALL(numDims, select_points_pd_gpu_kernel, 
            totalNumBlocks, blockSize, 0,
            numPts, numKeys, 
            (const int*)validCellTensor.data_ptr(), 
            (const int*)pGridSize.data_ptr(), 
            (const mccnn::int64_m*)std::get<0>(uniqueKeysTuple).data_ptr(), 
            (const int2*)pNeighbors.data_ptr(), 
            (const int*)pStartIds.data_ptr(), 
            (const int*)accumNumPts.data_ptr(),
            (int*)selected.data_ptr());
    }

    // Return the selected indices.
    return torch::nonzero(selected);
}
