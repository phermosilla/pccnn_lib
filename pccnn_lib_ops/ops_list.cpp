/////////////////////////////////////////////////////////////////////////////
/// \file ops_list.cpp
///
/// \brief Declaration of all operations in the module. 
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "./build_grid/build_grid_ds.cuh"
#include "./compute_keys/compute_keys.cuh"
#include "./find_neighbors/find_neighbors.cuh"
#include "./compute_pdf/compute_pdf.cuh"
#include "./poisson_disk_sampling/poisson_disk_sampling.cuh"

#include "./pcconvs/mcconv.cuh"
#include "./pcconvs/pointconv.cuh"
#include "./pcconvs/kpconv.cuh"
#include "./pcconvs/kpconvn.cuh"
#include "./pcconvs/sphconv.cuh"
#include "./pcconvs/pccnn.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_grid_ds", &build_grid_ds, "Build Grid Data Structure");
    m.def("compute_keys", &compute_keys, "Compute keys regular grid");
    m.def("find_neighbors", &find_neighbors, "Find neighbors");
    m.def("compute_pdf", &compute_pdf, "Compute pdf");
    m.def("compute_pdf_grads", &compute_pdf_grads, "Compute pdf grads");
    m.def("poisson_disk_sampling", &poisson_disk_sampling, "Poisson disk sampling");
    
    m.def("mc_conv", &mcconv, "McConv");
    m.def("mc_conv_grads", &mcconv_grads, "McConvGrads");
    m.def("mc_conv_weight_var", &mcconv_compute_weight_variance, "McConv weight variance");
    m.def("point_conv_basis", &pointconv_basis, "PointConv basis");
    m.def("point_conv_basis_grads", &pointconv_basis_grads, "PointConv basis grads");
    m.def("point_conv", &pointconv, "PointConv");
    m.def("point_conv_grads", &pointconv_grads, "PointConvGrads");
    m.def("point_conv_weight_var", &pointconv_compute_weight_variance, "PointConv weight variance");
    m.def("kp_conv", &kpconv, "KpConv");
    m.def("kp_conv_grads", &kpconv_grads, "KpConvGrads");
    m.def("kp_conv_weight_var", &kpconv_compute_weight_variance, "KPConv weight variance");
    m.def("kp_conv_n", &kpconvn, "KpConvN");
    m.def("kp_conv_n_grads", &kpconvn_grads, "KpConvNGrads");
    m.def("kp_conv_n_weight_var", &kpconvn_compute_weight_variance, "KPConvN weight variance");
    m.def("sph_conv", &sphconv, "SPHConv");
    m.def("sph_conv_grads", &sphconv_grads, "SPHConvGrads");
    m.def("sph_conv_weight_var", &sphconv_compute_weight_variance, "SPHConv weight variance");
    m.def("pccnn", &pccnn, "PCCNN");
    m.def("pccnn_grads", &pccnn_grads, "PCCNNGrads");
    m.def("pccnn_weight_variance", &pccnn_compute_weight_variance, "PCCNN weight variance");
}