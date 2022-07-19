/////////////////////////////////////////////////////////////////////////////
/// \file compute_variance_weights.cuh
///
/// \brief Implementation of the operations to compute the variance of the 
///     convolution weights.
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
#include "../../shared/gpu_device_utils.cuh"

#include "feat_basis_utils.cuh"
#include "compute_variance_weights.cuh"
    
///////////////////////// GPU

template<int K, typename scalar_t>
__global__ void compute_variance_weights_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumFeatures,
    const bool pDivideNumNeighs,
    const scalar_t* __restrict__ pInFeatures,
    const float* __restrict__ pPtsBasisProj,
    const int2* __restrict__ pNeighbors,
    const int* __restrict__ pNeighIndexXSample,
    float* __restrict__ pAccumMeanFeat)
{
    // Declare shared memory.
    extern __shared__ float sharedMemory[];

    // Get the global thread index.
    for(unsigned int curIter = blockIdx.x; 
        curIter < pNumSamples; 
        curIter += gridDim.x)
    {

        // Get the range of points for this receptive field.
        int2 rangePts;
        rangePts.x = (curIter > 0)?pNeighIndexXSample[curIter-1]:0;
        rangePts.y = pNeighIndexXSample[curIter];
        int numPts = rangePts.y-rangePts.x;
        float divNumNeighs = 1.0f/(float)(numPts);
        if(!pDivideNumNeighs)
            divNumNeighs = 1.0f;

        // Iterate over the points in the receptive field and compute the contribution.
        float accumMeanFeat = 0.0f;
        float accumMeanBasis = 0.0f;
        for(int i = 0; i < numPts; ++i)
        {
            // Get the neighbor indices.
            int2 neighIndex1 = pNeighbors[rangePts.x+i];
            
            // Store pt basis project in shared memory.
            for(int kiter = threadIdx.x; kiter < K; kiter+=blockDim.x)
                sharedMemory[kiter] = pPtsBasisProj[(rangePts.x+i)*K + kiter];

            __syncthreads();
            
            for(int j = 0; j < numPts; ++j)
            {
                // Get the neighbor indices.
                int2 neighIndex2 = pNeighbors[rangePts.x+j];

                //Compute the sum of multiplication of basis projection.
                float sumMulBasis = 0.0f;
#pragma unroll
                for(int kiter = 0; kiter < K; ++kiter)
                    sumMulBasis += sharedMemory[kiter]*
                        pPtsBasisProj[(rangePts.x+j)*K + kiter];

                // Accumulate.
                accumMeanBasis += sumMulBasis;
                
                //Iterate over the features and accumulate contributions.
                for(int f1 = threadIdx.x; f1 < pNumFeatures; f1 += blockDim.x)
                {
                    float feature1 = pInFeatures[neighIndex1.x*pNumFeatures + f1];
                    float feature2 = pInFeatures[neighIndex2.x*pNumFeatures + f1];

                    // Accumulate.
                    accumMeanFeat += (feature1*feature2*divNumNeighs*divNumNeighs*sumMulBasis);
                }
            } 

            __syncthreads();           
        }

        sharedMemory[K + threadIdx.x] = accumMeanFeat;

        __syncthreads();

        if(threadIdx.x == 0){
            float accumTotalMeanFeat = 0.0f;
            float accumTotalNumPts = numPts*numPts;
            for(int iter = 0; iter < blockDim.x; ++iter)
                accumTotalMeanFeat += sharedMemory[K + iter];
            pAccumMeanFeat[curIter] = accumTotalMeanFeat;
        }

        __syncthreads();
    }
}

///////////////////////// CPU

torch::Tensor compute_variance_weights(
    torch::Tensor& pPtBasis,
    torch::Tensor& pPtFeatures,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pStartIds,
    bool pDivideNumNeighs)
{
    // Get the number of dimensions and points.
    int numBasis = pPtBasis.size(1);
    int numPts = pStartIds.size(0);
    int numFeatures = pPtFeatures.size(1);

    // Get device properties.
    cudaDeviceProp props = mccnn::get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    FEAT_FUNCT_PTR(numBasis, pPtFeatures.scalar_type(), 
        compute_variance_weights_gpu_kernel, funcPtr);
        
    //Calculate the block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;

    //Calculate the shared memory needed.
    unsigned int sharedMemSize = (numBasis + blockSize)*sizeof(float);

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numBlocks = mccnn::get_max_active_block_x_sm(
        blockSize, funcPtr, sharedMemSize);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = numPts;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Create output.
    auto tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).
        device(pPtFeatures.device().type(), pPtFeatures.device().index());
    auto outTensorMeanFeat = torch::zeros({numPts}, tensorOptions);

    // Call the function.
    FEAT_FUNCT_CALL(numBasis, pPtFeatures.scalar_type(), 
        totalNumBlocks, blockSize, sharedMemSize,
        "compute_variance_weights_gpu_kernel", compute_variance_weights_gpu_kernel, 
            numPts, numFeatures, pDivideNumNeighs,
            (const scalar_t*)pPtFeatures.data_ptr(),
            (const float*)pPtBasis.data_ptr(), 
            (const int2*)pNeighbors.data_ptr(), 
            (const int*)pStartIds.data_ptr(), 
            (float*)outTensorMeanFeat.data_ptr()
        )

    // Calculate the variance.
    auto meanFeats = torch::mean(outTensorMeanFeat);
    return torch::reciprocal(meanFeats);
}
