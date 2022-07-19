/////////////////////////////////////////////////////////////////////////////
/// \file spherical_basis.cu
///
/// \brief Implementation of the CUDA operations to compute the projection of
///     the points into an mlp basis.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "../../shared/defines.cuh"
#include "../../shared/math_helper.cuh"
#include "../../shared/cuda_kernel_utils.cuh"
#include "../../shared/grid_utils.cuh"
#include "../../shared/gpu_device_utils.cuh"

#include "spherical_basis.cuh"
    
///////////////////////// GPU

__global__ void compute_spherical_basis_gpu_kernel(
    const unsigned int pNumNeighs,
    const unsigned int pNumDistances,
    const unsigned int pNumAzimuz,       
    const unsigned int pNumPolar,
    const float* __restrict__ pDistances,
    const float* __restrict__ pAzimuz,
    const float* __restrict__ pPolar,
    const float* __restrict__ pBinsDistances,
    const float* __restrict__ pBinsAzimuz,
    const float* __restrict__ pBinsPolar,
    float* __restrict__ pOutBasis)
{
    // Get the global thread index.
    int iniPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    // Calculate the number of basis.
    int numBasis = pNumDistances * pNumAzimuz * pNumPolar + 1;

    // Iterate over the points.
    for(unsigned int curIter = iniPtIndex; 
        curIter < pNumNeighs; 
        curIter += totalThreads)
    {
        // Get the values of the point.
        float curDist = pDistances[curIter];
        float curAzimuz = pAzimuz[curIter];
        float curPolar = pPolar[curIter];

        // Center multiplier.
        int centerId =  (curDist < 1e-4)?0:1;

        // Distance multiplier.
        int distId = -1;
//#pragma unroll(2)
        for(int i = 0; i < pNumDistances; ++i)
            if(curDist <= pBinsDistances[i] && distId < 0)
                distId = i;
        distId = (distId < 0)?pNumDistances-1:distId;
        
        // Azimuz multiplier.
        int azimuzId = -1;
//#pragma unroll(2)
        for(int i = 0; i < pNumAzimuz; ++i)
            if(curAzimuz <= pBinsAzimuz[i] && azimuzId < 0)
                azimuzId = i;
        azimuzId = (azimuzId < 0)?pNumAzimuz-1:azimuzId;

        // Polar multiplier.
        int polarId = -1;
//#pragma unroll(2)
        for(int i = 0; i < pNumPolar; ++i)
            if(curPolar <= pBinsPolar[i] && polarId < 0)
                polarId = i;
        polarId = (polarId < 0)?pNumPolar-1:polarId;

        // Save the result.
        int basisId = distId * pNumAzimuz * pNumPolar +
            azimuzId * pNumPolar + polarId + 1;
        basisId = basisId * centerId;
        pOutBasis[curIter*numBasis + basisId] = 1.0f;
    }
}

///////////////////////// CPU

torch::Tensor spherical_basis_gpu_kernel_wrapper(
    torch::Tensor& pRadial,
    torch::Tensor& pAzimuz,
    torch::Tensor& pPolar,
    torch::Tensor& pRadialBins,
    torch::Tensor& pAzimuzBins,    
    torch::Tensor& pPolarBins)
{
    // Get sizes.
    int numNeighbors = pRadial.size(0);
    int numRadialBins = pRadialBins.size(0);
    int numAzimuzBins = pAzimuzBins.size(0);
    int numPolarBins = pPolarBins.size(0);
    int numBasis = numRadialBins*numAzimuzBins*numPolarBins + 1;

    // Get device properties.
    cudaDeviceProp props = mccnn::get_cuda_device_properties();

    // Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;
    unsigned int numBlocks = mccnn::get_max_active_block_x_sm(
        blockSize, (const void*)compute_spherical_basis_gpu_kernel, 0);

    // Calculate the total number of blocks to execute.
    unsigned int execBlocks = numNeighbors/blockSize;
    execBlocks += (numNeighbors%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Create output.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).
        device(pRadial.device().type(), pRadial.device().index());
    auto outTensor = torch::zeros({numNeighbors, numBasis}, tensorOptions);
    //auto outTensor = torch::ones({numNeighbors, numBasis}, tensorOptions);
    //outTensor = outTensor*0.1;

    // Call the cuda kernel.
    compute_spherical_basis_gpu_kernel<<<totalNumBlocks, blockSize, 0>>>(
        numNeighbors, numRadialBins, numAzimuzBins, numPolarBins,
        (const float*)pRadial.data_ptr(),
        (const float*)pAzimuz.data_ptr(),
        (const float*)pPolar.data_ptr(),
        (const float*)pRadialBins.data_ptr(),
        (const float*)pAzimuzBins.data_ptr(),
        (const float*)pPolarBins.data_ptr(),
        (float*)outTensor.data_ptr());

    // Return result.
    return outTensor;
}

torch::Tensor spherical_basis(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pRadialBins,
    torch::Tensor& pAzimuzBins,    
    torch::Tensor& pPolarBins)
{
    // Compute distance.
    auto distances = torch::mul(pPtDiffs, pPtDiffs);
    distances = torch::sum(distances, -1);
    distances = torch::sqrt(distances);

    // Compute azimuz angle.
    auto azimuz = torch::atan(pPtDiffs.index({"...", 1})/(pPtDiffs.index({"...", 0})+1e-6));

    // Compute polar angle.
    auto acosInput = pPtDiffs.index({"...", 2})/(distances+1e-6);
    acosInput = torch::clamp(acosInput, -0.99999, 0.99999);
    auto polar = torch::acos(acosInput);

    // Call the gpu kernel.
    auto basis = spherical_basis_gpu_kernel_wrapper(
        distances, azimuz, polar,
        pRadialBins, pAzimuzBins, pPolarBins);

    // Return basis.
    return basis;    
}