/////////////////////////////////////////////////////////////////////////////
/// \file mlp_basis.cu
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

#include "mlp_basis.cuh"
    
///////////////////////// GPU


///////////////////////// CPU

torch::Tensor mlp_basis(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pAxisProj,    
    torch::Tensor& pBiasAxis,
    torch::Tensor& pPDFs,
    const bool pDiv)
{
    // Project into the axes.
    torch::Tensor pointProj = torch::matmul(pPtDiffs, pAxisProj);

    // Add the bias.
    pointProj = pointProj + pBiasAxis;

    // Leaky relu.
    pointProj = torch::leaky_relu(pointProj, 0.2);

    // Weight by the pdfs.
    if(pDiv)
        return pointProj / torch::reshape(pPDFs, {pointProj.size(0), 1});
    else
        return pointProj * torch::reshape(pPDFs, {pointProj.size(0), 1});
}

std::vector<torch::Tensor> mlp_basis_grads(
    torch::Tensor& pPtDiffs,
    torch::Tensor& pAxisProj,    
    torch::Tensor& pBiasAxis,
    torch::Tensor& pPDFs,
    torch::Tensor& pInGradients,
    const bool pDiv)
{
    // Project into the axes.
    torch::Tensor pointProj = torch::matmul(pPtDiffs, pAxisProj);

    // Add the bias.
    pointProj = pointProj + pBiasAxis;

    // Leaky relu.
    pointProj = torch::leaky_relu(pointProj, 0.2);

    // PDF gradients.
    torch::Tensor outPDFGradients;
    if(pDiv){
        outPDFGradients = (-1.0*pointProj*pInGradients)/torch::reshape(
            pPDFs*pPDFs, {pointProj.size(0), 1});
        outPDFGradients = torch::sum(outPDFGradients, 1);
    }else{
        outPDFGradients = pointProj*pInGradients;
        outPDFGradients = torch::sum(outPDFGradients, 1);
    }

    // Current gradient.
    auto curGradient = pInGradients/torch::reshape(pPDFs, {-1, 1});
    auto mask = (pointProj > 0.0).to(torch::kFloat32);
    mask = mask + (1.0 - mask)*0.2;
    curGradient = curGradient * mask;

    // Gradient bias.
    auto outAxisBiasGradients = torch::sum(curGradient, 0, true);

    // Gradient pt diffs.
    auto axisProjTranspose = torch::transpose(pAxisProj, 0, 1);
    auto outPtDiffGradients = torch::matmul(curGradient, axisProjTranspose);

    // Gradient axis.
    auto ptDiffsTranspose = torch::transpose(pPtDiffs, 0, 1);
    auto outAxisGradients = torch::matmul(ptDiffsTranspose, curGradient);

    return {outPtDiffGradients, outAxisGradients, outAxisBiasGradients, outPDFGradients};
}