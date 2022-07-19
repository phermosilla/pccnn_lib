/////////////////////////////////////////////////////////////////////////////
/// \file compute_pdf.cu
///
/// \brief Implementation of the CUDA operations to compute the pdf of each 
///     neighboring point. 
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
#include "../shared/gpu_device_utils.cuh"

#include "compute_pdf.cuh"


///////////////////////// GPU

template<int D>
__global__ void compute_pdf_gpu_kernel(
    const float pManifoldDims,
    const unsigned int pNumSamples,
    const float* __restrict__ pInvBandwidth,
    const float* __restrict__ pPts,
    const int2* __restrict__ pNeighbors,
    const int* __restrict__ pNeighIndexXSample,
    float* __restrict__ pOutPDF)
{
    // Get the pointers.
    mccnn::fpoint<D> curBandwidth(pInvBandwidth);
    const mccnn::fpoint<D>* ptsPtr = (const mccnn::fpoint<D>*)pPts;

    // Get the global thread index.
    int iniPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    for(unsigned int curIter = iniPtIndex; 
        curIter < pNumSamples; 
        curIter += totalThreads)
    {
        // Get the current point coordinates.
        mccnn::fpoint<D> curPt = ptsPtr[curIter];

        // Get the range of points for this receptive field.
        int2 rangePts;
        rangePts.x = (curIter > 0)?pNeighIndexXSample[curIter-1]:0;
        rangePts.y = pNeighIndexXSample[curIter];
        int numPts = rangePts.y-rangePts.x;

        // Iterate over the points in the receptive field and compute the PDF.
        float accumPdf = 0.0f;
        for(int i = 0; i < numPts; ++i)
        {
            // Get the neighbor coordinate.
            int2 neighIndex = pNeighbors[rangePts.x+i];

            // Comopute the contribution to the KDE.
            mccnn::fpoint<D> diffVec = (ptsPtr[neighIndex.x] - curPt)*curBandwidth;
            float auxDot = exp(mccnn::dot(diffVec, diffVec)*(-0.5f));
            float localPDF = 1.0f;
#pragma unroll
            for(int d = 0; d < D; ++d)
                localPDF *= curBandwidth[d];

            // Accumulate the contribution.
            accumPdf += (localPDF) * (1.0/pow(2.0*3.1415, pManifoldDims/2.0)) * auxDot;
        }

        // Save the PDF.
        pOutPDF[curIter] = accumPdf;
    }
}


 template<int D>
 __global__ void compute_pdf_grads_gpu_kernel(
    const float pManifoldDims,
    const unsigned int pNumSamples,
    const float* __restrict__ pInvBandwidth,
    const float* __restrict__ pPts,
    const int2* __restrict__ pNeighbors,
    const int* __restrict__ pNeighIndexXSample,
    const float* __restrict__ pPDFGrads,
    float* __restrict__ pOutPtGrads)
 {
     // Get the pointers.
    mccnn::fpoint<D> curBandwidth(pInvBandwidth);
    const mccnn::fpoint<D>* ptsPtr = (const mccnn::fpoint<D>*)pPts;
    mccnn::fpoint<D>* outGradsPtr = (mccnn::fpoint<D>*)pOutPtGrads;

    // Get the global thread index.
    int iniPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curIter = iniPtIndex; 
        curIter < pNumSamples; 
        curIter += totalThreads)
    { 
        // Get the current point coordinates.
        mccnn::fpoint<D> curPt = ptsPtr[curIter];
 
        // Get the range of points for this receptive field.
        int2 rangePts;
        rangePts.x = (curIter > 0)?pNeighIndexXSample[curIter-1]:0;
        rangePts.y = pNeighIndexXSample[curIter];
        int numPts = rangePts.y-rangePts.x;


        // Get the current pdf gradient.
        float curPDFGrad = pPDFGrads[curIter];

        // Iterate over the points in the receptive field and compute the PDF.
        mccnn::fpoint<D> accumGradient(0.0f);
        for(int i = 0; i < numPts; ++i)
        {
            // Get the neighbor coordinate.
            int2 neighIndex = pNeighbors[rangePts.x+i];

            // Comopute the contribution to the KDE.
            mccnn::fpoint<D> diffVec = (ptsPtr[neighIndex.x] - curPt)*curBandwidth;
            float auxDot = exp(mccnn::dot(diffVec, diffVec)*(-0.5f));
            float localPDF = 1.0f;
#pragma unroll
            for(int d = 0; d < D; ++d)
                localPDF *= curBandwidth[d];

            // Local pdf.
            localPDF = (localPDF) * (1.0/pow(2.0*3.1415, pManifoldDims/2.0)) * auxDot;

            // Local gradient.
            mccnn::fpoint<D> localGrad = diffVec*curBandwidth*curPDFGrad*localPDF;

            // Accumulate the contribution.
            accumGradient += localGrad;

#pragma unroll
            for(int d = 0; d < D; ++d)
                atomicAdd(&outGradsPtr[neighIndex.x][d], -localGrad[d]);
        }
#pragma unroll
        for(int d = 0; d < D; ++d)
            atomicAdd(&outGradsPtr[curIter][d], accumGradient[d]);
    }
 }

///////////////////////// CPU

torch::Tensor compute_pdf(
    const int pManifoldDims,
    const torch::Tensor& pBandWidth,
    const torch::Tensor& pPts,
    const torch::Tensor& pNeighbors,
    const torch::Tensor& pStartI)
{
    // Get the number of dimensions and points.
    int numDims = pPts.size(1);
    int numPts = pPts.size(0);
    int numNeighbors = pNeighbors.size(0);

    // Get device properties.
    cudaDeviceProp props = mccnn::get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    DIMENSION_FUNCT_PTR(numDims, compute_pdf_gpu_kernel, funcPtr);

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
    auto tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).
        device(pPts.device().type(), pPts.device().index());
    auto outTensor = torch::zeros({numPts}, tensorOptions);

    // Compute inverse of the cell size.
    auto invBadnwidth = torch::reciprocal(pBandWidth);

    // Call the cuda kernel.
    DIMENSION_SWITCH_CALL(numDims, 
        compute_pdf_gpu_kernel, totalNumBlocks, blockSize, 0,
        (float)pManifoldDims, numPts, 
        (const float*)invBadnwidth.data_ptr(), 
        (const float*)pPts.data_ptr(), 
        (const int2*)pNeighbors.data_ptr(), 
        (const int*)pStartI.data_ptr(), 
        (float*)outTensor.data_ptr());

    return outTensor;
}


torch::Tensor compute_pdf_grads(
    const int pManifoldDims,
    const torch::Tensor& pBandWidth,
    const torch::Tensor& pPts,
    const torch::Tensor& pNeighbors,
    const torch::Tensor& pStartI,
    const torch::Tensor& pInGrads)
{
    // Get the number of dimensions and points.
    int numDims = pPts.size(1);
    int numPts = pPts.size(0);
    int numNeighbors = pNeighbors.size(0);

    // Get device properties.
    cudaDeviceProp props = mccnn::get_cuda_device_properties();

    // Get the function pointer.
    void* funcPtr = nullptr;
    DIMENSION_FUNCT_PTR(numDims, compute_pdf_grads_gpu_kernel, funcPtr);

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
    auto tensorOptions = torch::TensorOptions().dtype(torch::kFloat32).
        device(pPts.device().type(), pPts.device().index());
    auto outTensor = torch::zeros({numPts, numDims}, tensorOptions);

    // Compute inverse of the cell size.
    auto invBadnwidth = torch::reciprocal(pBandWidth);

    // Call the cuda kernel.
    DIMENSION_SWITCH_CALL(numDims, compute_pdf_grads_gpu_kernel, totalNumBlocks, blockSize, 0,
        (float)pManifoldDims, numPts, 
        (const float*)invBadnwidth.data_ptr(), 
        (const float*)pPts.data_ptr(), 
        (const int2*)pNeighbors.data_ptr(), 
        (const int*)pStartI.data_ptr(), 
        (const float*)pInGrads.data_ptr(),
        (float*)outTensor.data_ptr());

    return outTensor;
}
