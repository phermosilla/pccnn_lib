import os
import sys
import time
import math
import numpy as np
from sklearn.neighbors import KernelDensity
import warnings
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib
from pccnn_lib.op_wrappers import ComputeKPConv

def test_kpconv_grads(p_config_dict, p_data):
    
    # Get the parameters.
    rnd_seed = int(p_config_dict['rnd_seed'])
    batch_size = int(p_config_dict['batch_size'])
    num_pts = int(p_config_dict['num_pts'])
    radius = float(p_config_dict['radius'])
    num_dims = int(p_config_dict['num_dims'])
    num_in_features = int(p_config_dict['num_in_features'])
    num_out_features = int(p_config_dict['num_out_features'])
    num_kernel_pts = int(p_config_dict['num_kernel_pts'])

    # Get the points.
    selected_points = p_data[0:batch_size, 0:num_pts, :]
    selected_points = selected_points.reshape((-1, 3))
    if num_dims < 3:
        selected_points = selected_points[:, 0:num_dims]
    elif num_dims > 3:
        selected_points = np.concatenate([
            selected_points[:, 0:3],
            selected_points[:, 0:num_dims-3]],
            axis=-1)

    # Get the batch ids.
    batch_ids = np.arange(batch_size).reshape((batch_size, 1))
    batch_ids = np.repeat(batch_ids, num_pts, 1).reshape((-1)).astype(np.int32)

    # Get the features.
    np.random.seed(rnd_seed)
    features = np.random.normal(0.0, 1.0, size=(num_pts*batch_size, num_in_features))
    features = features.astype(np.float32)

    # Bandwidth.
    radius_np = np.array([radius for i in range(num_dims)], 
        dtype=np.float32)

    # Weights.
    kpconv_layer = pccnn_lib.pc.layers.KPConv(
        num_dims, num_in_features, num_out_features, num_kernel_pts)
    kpconv_layer.cuda()
    

    # Define pytorch tensors.
    device = torch.device("cuda:0")
    torch_pts = torch.as_tensor(selected_points, device=device)
    torch_features = torch.as_tensor(features, device=device)
    torch_kernel_pts = torch.as_tensor(kpconv_layer.kernel_pts_, device=device)
    torch_conv_weights = torch.as_tensor(kpconv_layer.conv_weights_, device=device)
    torch_pts.requires_grad = True
    torch_features.requires_grad = True
    torch_kernel_pts.requires_grad = True
    torch_conv_weights.requires_grad = True

    # Get sigma value.
    sigma = kpconv_layer.sigma_.item()

    # Create the point cloud.
    pc = pccnn_lib.pc.Pointcloud(torch_pts, batch_ids, device=device)

    # Create the radius tensor.
    radius = torch.as_tensor(radius_np, device=device)

    # Compute neighborhood.
    bb = pccnn_lib.pc.BoundingBox(pc)
    grid = pccnn_lib.pc.Grid(pc, bb, radius)
    grid.compute_cell_ids()
    grid.build_ds()
    neighborhood = pccnn_lib.pc.Neighborhood(grid, pc, 0)

    # Definition of functions to measure the gradients.
    def compute_conv_features(features):
        out_features = ComputeKPConv.apply(
            pc.pts_, features, pc.pts_,
            neighborhood.neighbors_, neighborhood.start_ids_, radius,
            torch_kernel_pts, sigma, torch_conv_weights)
        return torch.mean(out_features)

    def compute_conv_kernel_pts(kernel_pts):
        out_features = ComputeKPConv.apply(
            pc.pts_, torch_features, pc.pts_,
            neighborhood.neighbors_, neighborhood.start_ids_, radius,
            kernel_pts, sigma, torch_conv_weights)
        return torch.mean(out_features)

    def compute_conv_proj_weights(conv_weights):
        out_features = ComputeKPConv.apply(
            pc.pts_, torch_features, pc.pts_,
            neighborhood.neighbors_, neighborhood.start_ids_, radius,
            torch_kernel_pts, sigma, conv_weights)
        return torch.mean(out_features)

    # Use torch modules.
    start_time = current_milli_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        test = torch.autograd.gradcheck(compute_conv_proj_weights, 
            torch_conv_weights, eps=1e-3, atol = 1e-3, rtol = 1e-3, 
            nondet_tol = 1e-3, raise_exception=True)

        test = test and torch.autograd.gradcheck(compute_conv_features, 
            torch_features, eps=1e-3, atol = 1e-3, rtol = 1e-3, 
            nondet_tol = 1e-3, raise_exception=True)

        test = test and torch.autograd.gradcheck(compute_conv_kernel_pts, 
            torch_kernel_pts, eps=1e-4, atol = 1e-1, rtol = 1e-1, 
             nondet_tol = 1e-1, raise_exception=True)

    end_time = current_milli_time()

    return test, end_time- start_time