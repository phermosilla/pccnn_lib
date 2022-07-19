/////////////////////////////////////////////////////////////////////////////
/// \file gpu_device_utils.h
///
/// \brief Utils of gpu devices.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _GPU_DEVICE_UTILS_H_
#define _GPU_DEVICE_UTILS_H_

#include "cuda_runtime.h"
#include "defines.cuh"

namespace mccnn{

    /**
     *  Method to get the properties of the device.
     *  @return Cuda device properties.
     */
     __forceinline__ cudaDeviceProp get_cuda_device_properties()
    {
        cudaDeviceProp prop;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);

        return prop;
    };


    /**
     *  Get the maximum number of active blocks per sm.
     *  @param  pBlockSize          Size of the block.
     *  @param  pKernel             Kernel.
     *  @param  pSharedMemXBlock    Dynamic shared memory per block.
     */
     __forceinline__ int get_max_active_block_x_sm(
                const unsigned int pBlockSize, 
                const void* pKernel,
                const size_t pSharedMemXBlock)
    {
        int outputNumBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor ( 
            &outputNumBlocks, pKernel, pBlockSize, pSharedMemXBlock);
        return outputNumBlocks;
    };
}

#endif