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

#include "../shared/defines.cuh"
#include "../shared/math_helper.cuh"
#include "../shared/cuda_kernel_utils.cuh"
#include "../shared/grid_utils.cuh"
#include "../shared/gpu_device_utils.cuh"

#include "find_neighbors.cuh"
#include "find_ranges_grid_ds.cuh"
#include "count_neighbors.cuh"
#include "store_neighbors.cuh"

///////////////////////// CPU

std::vector<torch::Tensor> find_neighbors(
    torch::Tensor pPts,
    torch::Tensor pPtsKeys,
    torch::Tensor pSample,
    torch::Tensor pSampleKeys,
    torch::Tensor pGridDS,
    torch::Tensor pGridSize,
    torch::Tensor pRadii,
    unsigned int pMaxNeighbors) {

    // Get the number of dimensions and points.
    int numDims = pGridSize.size(0);
    int numPts = pPtsKeys.size(0);
    int numSamples = pSampleKeys.size(0);

    // Compute the number of offsets to used in the search.
    std::vector<int> combOffsets;
    unsigned int numOffsets = computeTotalNumOffsets(
        numDims, 1, combOffsets);

    // Upload offsets to gpu.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kInt32);
    torch::Device cudaDevice(pPtsKeys.device().type(), pPtsKeys.device().index());
    auto offsetTensor = torch::from_blob(&combOffsets[0], 
        {numOffsets, numDims}, tensorOptions).to(cudaDevice);

    // Find ranges.
    auto outRanges = find_ranges_grid_ds_gpu(
        1, offsetTensor, pSampleKeys, pPtsKeys,
        pGridSize, pGridDS);

    // Count neighbors.
    auto invRadii = torch::reciprocal(pRadii);
    auto counts = count_neighbors(pSample, pPts, outRanges, invRadii);

    // Store neighbors and return result.
    return store_neighbors(
        pMaxNeighbors, pSample, pPts,
        outRanges, invRadii, counts);
}
