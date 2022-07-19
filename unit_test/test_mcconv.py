import os
import sys
import time
import numpy as np
from sklearn.neighbors import KernelDensity
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_mcconv(p_config_dict, p_data):
    
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
    features = np.random.uniform(low=0.0, high=1.0, size=(num_pts*batch_size, num_in_features))
    features = features.astype(np.float32)

    # Bandwidth.
    radius_np = np.array([radius for i in range(num_dims)], 
        dtype=np.float32)
    badnwidth_np = np.array([bandwidth for i in range(num_dims)], 
        dtype=np.float32)

    # Use torch modules.
    device = torch.device("cuda:0")
    pc = pccnn_lib.pc.Pointcloud(selected_points, batch_ids, device=device)
    feats = torch.as_tensor(features, device=device)

    start_time = current_milli_time()
    pc.compute_pdf(badnwidth_np)
    mcconv_layer = pccnn_lib.pc.layers.MCConv(
        num_dims, num_in_features, num_out_features, num_hidden)
    mcconv_layer.cuda()
    out_conv, neighborhood = mcconv_layer(pc, pc, feats, radius_np)
    end_time = current_milli_time()

    # Numpy.
    np_neighbors = neighborhood.neighbors_.cpu().numpy()
    np_pt_neighs = selected_points[np_neighbors[:, 0], :]
    np_sample_neighs = selected_points[np_neighbors[:, 1], :]
    np_diff_pts = (np_pt_neighs-np_sample_neighs)/radius_np.reshape((-1, num_dims))    
    np_pdfs = pc.pts_unnorm_pdf_.cpu().numpy()*np.prod(radius_np)

    np_axis_weights = mcconv_layer.proj_axis_.cpu().detach().numpy()
    np_axis_bias = mcconv_layer.proj_bias_.cpu().detach().numpy()
    np_axis_proj = np.dot(np_diff_pts, np_axis_weights) + np_axis_bias
    np_axis_proj = np_axis_proj.reshape((-1))
    np_neg_mask = np_axis_proj <  0.0
    np_axis_proj[np_neg_mask] = (np_axis_proj[np_neg_mask]*0.2)
    np_axis_proj = np_axis_proj.reshape((-1, num_hidden))
    cur_pdfs = np_pdfs[np_neighbors[:, 0]]
    np_axis_proj = np_axis_proj / cur_pdfs.reshape((-1, 1))

    np_proj_feats = np.zeros((selected_points.shape[0], num_in_features, num_hidden))
    for np_neigh_iter in range(np_neighbors.shape[0]):
        cur_neigh_ids = np_neighbors[np_neigh_iter, :]
        cur_neigh_feats = features[cur_neigh_ids[0], :].reshape((-1, 1))
        cur_projs = np_axis_proj[np_neigh_iter, :].reshape((1, -1))
        cur_feats = cur_neigh_feats * cur_projs
        np_proj_feats[cur_neigh_ids[1], :, :] += cur_feats
    np_proj_feats = np_proj_feats.reshape((selected_points.shape[0], -1))

    conv_weights = mcconv_layer.conv_weights_.cpu().detach().numpy().reshape((num_in_features*num_hidden, -1))
    conv_out_feats = np.dot(np_proj_feats, conv_weights)
    
    torch_conv_out_feats = out_conv.cpu().detach().numpy()

    equal = np.isclose(conv_out_feats,torch_conv_out_feats , rtol=1e-04, atol=1e-04)

    return equal.all(), end_time- start_time