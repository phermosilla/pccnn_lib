/////////////////////////////////////////////////////////////////////////////
/// \file spherical_basis.cuh
///
/// \brief Declaraion of the CUDA operations to compute the projection of
///     the points into an spherical basis.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef _SPHERICAL_BASIS_CUH_
#define _SPHERICAL_BASIS_CUH_

#include "../../shared/defines.cuh"
    
torch::Tensor spherical_basis(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pRadialBins,
    torch::Tensor& pAzimuzBins,    
    torch::Tensor& pPolarBins);


#endif