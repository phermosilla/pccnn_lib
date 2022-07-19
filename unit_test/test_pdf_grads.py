import os
import sys
import time
import numpy as np
from sklearn.neighbors import KernelDensity
import warnings
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_pdf_grads(p_config_dict, p_data):
    
    # Get the parameters.
    batch_size = int(p_config_dict['batch_size'])
    num_pts = int(p_config_dict['num_pts'])
    radius = float(p_config_dict['radius'])
    num_dims = int(p_config_dict['num_dims'])

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

    # Bandwidth.
    badnwidth = np.array([radius for i in range(num_dims)], 
        dtype=np.float32)
    
    # Define pytorch tensors.
    device = torch.device("cuda:0")
    torch_pts = torch.as_tensor(selected_points, device=device)
    torch_pts.requires_grad = True
    torch_batch_ids = torch.as_tensor(batch_ids, device=device)

    # Definition of function to compute the pdf.
    def compute_pdf(points):
        point_cloud = pccnn_lib.pc.Pointcloud(points, torch_batch_ids, 
            requires_grad=True, device=device)
        point_cloud.compute_pdf(badnwidth)
        return torch.mean(point_cloud.pts_pdf_)

    # Use torch modules.
    start_time = current_milli_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = torch.autograd.gradcheck(compute_pdf, torch_pts, eps=1e-3,
            atol = 1e-2, rtol = 1e-2, nondet_tol = 1e-2, raise_exception=True)
    end_time = current_milli_time()

    return test, end_time- start_time