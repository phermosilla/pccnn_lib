from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pccnn_lib',
    ext_modules=[
        CUDAExtension(
            name = 'pccnn_lib_ops', 
            sources = [
            'pccnn_lib_ops/build_grid/build_grid_ds.cu',
            'pccnn_lib_ops/compute_keys/compute_keys.cu',
            'pccnn_lib_ops/find_neighbors/find_neighbors.cu',
            'pccnn_lib_ops/find_neighbors/find_ranges_grid_ds.cu',
            'pccnn_lib_ops/find_neighbors/count_neighbors.cu',
            'pccnn_lib_ops/find_neighbors/store_neighbors.cu',
            'pccnn_lib_ops/compute_pdf/compute_pdf.cu',
            'pccnn_lib_ops/poisson_disk_sampling/poisson_disk_sampling.cu',
            'pccnn_lib_ops/pcconvs/shared/pt_diff.cu',
            'pccnn_lib_ops/pcconvs/shared/mlp_basis.cu',
            'pccnn_lib_ops/pcconvs/shared/spherical_basis.cu',
            'pccnn_lib_ops/pcconvs/shared/kernel_pts_basis.cu',
            'pccnn_lib_ops/pcconvs/shared/feat_basis_proj.cu',
            'pccnn_lib_ops/pcconvs/shared/feat_basis_proj_grads.cu',
            'pccnn_lib_ops/pcconvs/shared/compute_variance_weights.cu',
            'pccnn_lib_ops/pcconvs/mcconv.cu',
            'pccnn_lib_ops/pcconvs/pointconv.cu',
            'pccnn_lib_ops/pcconvs/kpconv.cu',
            'pccnn_lib_ops/pcconvs/kpconvn.cu',
            'pccnn_lib_ops/pcconvs/sphconv.cu',
            'pccnn_lib_ops/pcconvs/pccnn.cu',
            'pccnn_lib_ops/ops_list.cpp'
            ])],
    cmdclass={'build_ext': BuildExtension},
    packages=[
            'pccnn_lib',
            'pccnn_lib.op_wrappers',
            'pccnn_lib.pc', 
            'pccnn_lib.pc.layers', 
            'pccnn_lib.pc.layers.norm_layers', 
            'pccnn_lib.pc.layers.kernel_points', 
            'pccnn_lib.pc.models', 
            'pccnn_lib.pc.blocks',
            'pccnn_lib.py_utils'],
    package_data={'pccnn_lib_ops': ['*.so']}
)
