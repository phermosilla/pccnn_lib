/////////////////////////////////////////////////////////////////////////////
/// \file feat_basis_proj.cu
///
/// \brief Implementation of the CUDA operations to compute the projection of
///     the features into the basis.
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

#include "feat_basis_proj.cuh"
#include "feat_basis_utils.cuh"
    
///////////////////////// GPU

//Definition of the number of which the number of features should be 
// multiple of.
#define MULTIPLE_IN_FEATURES 8

//WARNING - Group features should be equal or smaller than K.
template<int K, typename scalar_t>
__global__ void compute_weighted_in_features(
    const unsigned int pGroupFeatures,
    const unsigned int pNumSamples,       
    const unsigned int pNumInFeatures,
    const float* __restrict__ pInPtProjBasisGPUPtr,
    const int2* __restrict__ pInNeighborsGPUPtr,
    const int* __restrict__ pSampleNeighIdsGPUPtr,
    const scalar_t* __restrict__ pInFeaturesGPUPts,
    scalar_t* __restrict__ pOutProjFeatGPUPtr)
{
    extern __shared__ unsigned char sharedMemory[];
    scalar_t* sharedMemoryBuff = reinterpret_cast<scalar_t*>(sharedMemory);

    //Get the pointers to shared memory.
    scalar_t* accumWeightFeatures = sharedMemoryBuff;
    scalar_t* features = &sharedMemoryBuff[blockDim.x*pGroupFeatures];

    //Compute the total number of blocks executed and other
    //useful indices.
    unsigned int numGroupsXBlock = blockDim.x/K;
    unsigned int numFeatureBlocks = pNumInFeatures/pGroupFeatures;
    unsigned int localId = threadIdx.x%K;
    unsigned int groupId = threadIdx.x/K;
    unsigned int totalBlocks = pNumSamples*numFeatureBlocks;

    for(int curIter = blockIdx.x; 
        curIter < totalBlocks; 
        curIter += gridDim.x)
    {
        //Get the sample id and the feature offset.
        int sampleId = curIter/numFeatureBlocks;
        int featureOffset = (curIter%numFeatureBlocks)*pGroupFeatures;

        //Get the range of points for this receptive field.
        int2 rangePts;
        rangePts.x = (sampleId > 0)?pSampleNeighIdsGPUPtr[sampleId-1]:0;
        rangePts.y = pSampleNeighIdsGPUPtr[sampleId];
        int numNeighbors = rangePts.y - rangePts.x;
        numNeighbors += numGroupsXBlock-(numNeighbors%numGroupsXBlock);

        //Initialize shared memory.
#pragma unroll(8)
        for(int featIter = 0; featIter < pGroupFeatures; ++featIter)
            accumWeightFeatures[featIter*blockDim.x + threadIdx.x] = 0.0f;

        //Iterate over the neighbors.
        for(int curNeighIter = groupId; 
            curNeighIter < numNeighbors; 
            curNeighIter += numGroupsXBlock)
        {
            int neighIndex = curNeighIter+rangePts.x;
            float curWeight = 0.0;

            if(neighIndex < rangePts.y){
                //Get the neighbor index.
                int2 neighAndSampleIndices = pInNeighborsGPUPtr[neighIndex];

                //Save the weights in shared memory.
                curWeight = pInPtProjBasisGPUPtr[neighIndex*K + localId];

                //Save the features in shared memory.
                if(localId < pGroupFeatures)
                    features[groupId*pGroupFeatures + localId] = pInFeaturesGPUPts[
                        neighAndSampleIndices.x*pNumInFeatures 
                        + featureOffset + localId];
            }else if(localId < pGroupFeatures){
                features[groupId*pGroupFeatures + localId] = 0.0f;
            }

            __syncthreads();

            //Iterate over the features.
            //TODO - Kahan summation. However performance drops by half.
#pragma unroll(8)
            for(int featIter = 0; featIter < pGroupFeatures; ++featIter)
                accumWeightFeatures[featIter*blockDim.x + threadIdx.x] += 
                    features[groupId*pGroupFeatures + featIter]*curWeight;

            __syncthreads();
        }

        //Save the result.
        if(threadIdx.x < K){
            for(int featIter = 0; featIter < pGroupFeatures; ++featIter){
                float accumContribs = 0.0f;
#pragma unroll(4)
                for(int groupIter = 0; groupIter < numGroupsXBlock; ++groupIter){
                    accumContribs += accumWeightFeatures[featIter*blockDim.x + 
                        localId + groupIter*K];
                }

                pOutProjFeatGPUPtr[sampleId*pNumInFeatures*K + (featureOffset + featIter)*K 
                    + localId] = accumContribs;
            }
        }

        __syncthreads();
    }
}

///////////////////////// CPU

torch::Tensor feat_basis_proj(
    torch::Tensor& pPtBasis,
    torch::Tensor& pPtFeatures,    
    torch::Tensor& pNeighbors,
    torch::Tensor& pStartIds)
{
    // Get the number of dimensions and points.
    int numBasis = pPtBasis.size(1);
    int numPts = pStartIds.size(0);
    int numNeighbors = pNeighbors.size(0);
    int numFeatures = pPtFeatures.size(1);

    //Determine the group of features.
    unsigned int groupFeatSize = min(MULTIPLE_IN_FEATURES, numFeatures);

    // Get device properties.
    cudaDeviceProp props = mccnn::get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    FEAT_FUNCT_PTR(numBasis, pPtFeatures.scalar_type(), 
        compute_weighted_in_features, funcPtr);
        
    //Calculate the block size.
    unsigned int numMP = props.multiProcessorCount;
    unsigned int blockSize = props.warpSize*2;

    //Calculate the shared memory needed.
    unsigned int sharedMemSize = (blockSize*(groupFeatSize+1))*torch::elementSize(pPtFeatures.scalar_type());

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numBlocks = mccnn::get_max_active_block_x_sm(
        blockSize, funcPtr, sharedMemSize);

    //Calculate the total number of blocks to execute.
    unsigned int numFeatureBlocks = numFeatures/groupFeatSize;
    unsigned int execBlocks = numPts*numFeatureBlocks;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    // Create output.
    auto tensorOptions = torch::TensorOptions().dtype(pPtFeatures.scalar_type()).
        device(pPtFeatures.device().type(), pPtFeatures.device().index());
    auto outTensor = torch::zeros({numPts, numFeatures, numBasis}, tensorOptions);

    // Call the function.
    FEAT_FUNCT_CALL(numBasis, pPtFeatures.scalar_type(), totalNumBlocks, blockSize, sharedMemSize,
        "compute_weighted_in_features", compute_weighted_in_features, 
            groupFeatSize, numPts, numFeatures,
            (const float*)pPtBasis.data_ptr(), 
            (const int2*)pNeighbors.data_ptr(), 
            (const int*)pStartIds.data_ptr(), 
            (const scalar_t*)pPtFeatures.data_ptr(), 
            (scalar_t*)outTensor.data_ptr()
        )


    return outTensor;
}
