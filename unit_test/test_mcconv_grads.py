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
from pccnn_lib.op_wrappers import ComputeMCConv

def test_mcconv_grads(p_config_dict, p_data):
    
    # Get the parameters.
    rnd_seed = int(p_config_dict['rnd_seed'])
    batch_size = int(p_config_dict['batch_size'])
    num_pts = int(p_config_dict['num_pts'])
    radius = float(p_config_dict['radius'])
    bandwidth = float(p_config_dict['bandwidth'])
    num_dims = int(p_config_dict['num_dims'])
    num_in_features = int(p_config_dict['num_in_features'])
    num_out_features = int(p_config_dict['num_out_features'])
    num_hidden = int(p_config_dict['num_hidden'])

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
    badnwidth_np = np.array([bandwidth for i in range(num_dims)], 
        dtype=np.float32)

    # Weights.
    np_proj_axis = np.random.normal(0.0, 1.0, (
            num_dims, num_hidden)).astype(np.float32)
    np_proj_bias = np.zeros((1, num_hidden)).astype(np.float32)
    np_conv_weights = np.random.normal(0.0, math.sqrt(2.0/(num_hidden * num_in_features)), (
            1, num_hidden * num_in_features,
            num_out_features)).astype(np.float32)

    # Define pytorch tensors.
    device = torch.device("cuda:0")
    torch_pts = torch.as_tensor(selected_points, device=device)
    torch_features = torch.as_tensor(features, device=device)
    torch_proj_axis = torch.as_tensor(np_proj_axis, device=device)
    torch_proj_bias = torch.as_tensor(np_proj_bias, device=device)
    torch_conv_weights = torch.as_tensor(np_conv_weights, device=device)
    torch_pts.requires_grad = True
    torch_features.requires_grad = True
    torch_proj_axis.requires_grad = True
    torch_proj_bias.requires_grad = True
    torch_conv_weights.requires_grad = True

    # Create the point cloud.
    pc = pccnn_lib.pc.Pointcloud(torch_pts, batch_ids, device=device)
    pc.compute_pdf(badnwidth_np)

    # Create the radius tensor.
    radius = torch.as_tensor(radius_np, device=device)

    # Compute neighborhood.
    bb = pccnn_lib.pc.BoundingBox(pc)
    grid = pccnn_lib.pc.Grid(pc, bb, radius)
    grid.compute_cell_ids()
    grid.build_ds()
    neighborhood = pccnn_lib.pc.Neighborhood(grid, pc, 0)

    # Compute the pdfs.
    mul_val = torch.prod(radius)
    cur_pdf = pc.pts_unnorm_pdf_ * mul_val

    # Definition of functions to measure the gradients.
    def compute_conv_features(features):
        out_features = ComputeMCConv.apply(
            pc.pts_, cur_pdf, features, pc.pts_,
            neighborhood.neighbors_, neighborhood.start_ids_, radius,
            torch_proj_axis, torch_proj_bias, torch_conv_weights)
        return torch.mean(out_features)

    def compute_conv_proj_axis(proj_axis):
        out_features = ComputeMCConv.apply(
            pc.pts_, cur_pdf, torch_features, pc.pts_,
            neighborhood.neighbors_, neighborhood.start_ids_, radius,
            proj_axis, torch_proj_bias, torch_conv_weights)
        return torch.mean(out_features)

    def compute_conv_proj_bias(proj_bias):
        out_features = ComputeMCConv.apply(
            pc.pts_, cur_pdf, torch_features, pc.pts_,
            neighborhood.neighbors_, neighborhood.start_ids_, radius,
            torch_proj_axis, proj_bias, torch_conv_weights)
        return torch.mean(out_features)

    def compute_conv_proj_weights(conv_weights):
        out_features = ComputeMCConv.apply(
            pc.pts_, cur_pdf, torch_features, pc.pts_,
            neighborhood.neighbors_, neighborhood.start_ids_, radius,
            torch_proj_axis, torch_proj_bias, conv_weights)
        return torch.mean(out_features)

    # Use torch modules.
    start_time = current_milli_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        test = torch.autograd.gradcheck(compute_conv_proj_weights, 
            torch_conv_weights, eps=1e-3, atol = 1e-4, rtol = 1e-4, 
            nondet_tol = 1e-4, raise_exception=True)

        test = test and torch.autograd.gradcheck(compute_conv_features, 
            torch_features, eps=1e-3, atol = 1e-4, rtol = 1e-4, 
            nondet_tol = 1e-4, raise_exception=True)

        test = test and torch.autograd.gradcheck(compute_conv_proj_axis, 
            torch_proj_axis, eps=1e-3, atol = 1e-2, rtol = 1e-2, 
             nondet_tol = 1e-2, raise_exception=True)

        test = test and torch.autograd.gradcheck(compute_conv_proj_bias, 
            torch_proj_bias, eps=1e-3, atol = 1e-2, rtol = 1e-2, 
             nondet_tol = 1e-2, raise_exception=True)

    end_time = current_milli_time()

    return test, end_time- start_time