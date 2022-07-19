import os
import sys
import time
import numpy as np
from sklearn.neighbors import KernelDensity
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_kpconv(p_config_dict, p_data):
    
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
    features = np.random.uniform(low=0.0, high=1.0, size=(num_pts*batch_size, num_in_features))
    features = features.astype(np.float32)

    radius_np = np.array([radius for i in range(num_dims)], 
        dtype=np.float32)

    # Use torch modules.
    device = torch.device("cuda:0")
    pc = pccnn_lib.pc.Pointcloud(selected_points, batch_ids, device=device)
    feats = torch.as_tensor(features, device=device)

    start_time = current_milli_time()
    kpconv_layer = pccnn_lib.pc.layers.KPConv(
        num_dims, num_in_features, num_out_features, num_kernel_pts)
    kpconv_layer.cuda()
    out_conv, neighborhood = kpconv_layer(pc, pc, feats, radius_np)
    end_time = current_milli_time()

    # Numpy.
    np_neighbors = neighborhood.neighbors_.cpu().numpy()
    np_pt_neighs = selected_points[np_neighbors[:, 0], :]
    np_sample_neighs = selected_points[np_neighbors[:, 1], :]
    np_diff_pts = (np_pt_neighs-np_sample_neighs)/radius_np.reshape((-1, num_dims))    

    np_kernel_pts = kpconv_layer.kernel_pts_.cpu().detach().numpy()
    np_sigma = kpconv_layer.sigma_cpu_

    np_kernel_diff = np_diff_pts.reshape((-1, 1, num_dims)) - np_kernel_pts.reshape((1, -1, num_dims))
    np_kernel_dist = np.linalg.norm(np_kernel_diff, axis=-1)/np_sigma
    np_basis_proj = np.clip(1.0 - np_kernel_dist, 0.0, 1.0)

    np_proj_feats = np.zeros((selected_points.shape[0], num_in_features, num_kernel_pts))
    for np_neigh_iter in range(np_neighbors.shape[0]):
        cur_neigh_ids = np_neighbors[np_neigh_iter, :]
        cur_neigh_feats = features[cur_neigh_ids[0], :].reshape((-1, 1))
        cur_projs = np_basis_proj[np_neigh_iter, :].reshape((1, -1))
        cur_feats = cur_neigh_feats * cur_projs
        np_proj_feats[cur_neigh_ids[1], :, :] += cur_feats
    np_proj_feats = np_proj_feats.reshape((selected_points.shape[0], -1))

    conv_weights = kpconv_layer.conv_weights_.cpu().detach().numpy()
    conv_out_feats = np.dot(np_proj_feats, conv_weights)
    
    torch_conv_out_feats = out_conv.cpu().detach().numpy()
    equal = np.isclose(conv_out_feats,torch_conv_out_feats , rtol=1e-04, atol=1e-04)

    return equal.all(), end_time- start_time