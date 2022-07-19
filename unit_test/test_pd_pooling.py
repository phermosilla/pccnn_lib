import os
import sys
import time
import numpy as np
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_pd_pooling(p_config_dict, p_data):
    
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

    # Cell size.
    cell_size = np.array([radius for i in range(num_dims)], 
        dtype=np.float32)

    # Use torch modules.
    device = torch.device("cuda:0")
    pc = pccnn_lib.pc.Pointcloud(selected_points, batch_ids, device=device)
    start_time = current_milli_time()

    pt_pooling = pccnn_lib.pc.PointPooling(pc, cell_size)
    pool_pts = pt_pooling.pool_tensor(pc.pts_)
    pool_batch_ids = pt_pooling.pool_tensor(pc.batch_ids_)

    end_time = current_milli_time()
    torch_pool_pts = pool_pts.to(torch.float32).cpu().numpy()
    torch_batch_ids = pool_batch_ids.to(torch.int32).cpu().numpy()
    # Check minimum distance.
    valid_pts = True
    for cur_pt_iter, cur_pt in enumerate(torch_pool_pts):
        cur_batch_id = torch_batch_ids[cur_pt_iter]
        torch_batch_ids[cur_pt_iter] = -1
        diff_pts = cur_pt.reshape((1, num_dims)) - torch_pool_pts
        distances = np.sqrt(np.sum(diff_pts*diff_pts, axis=-1))
        mask = np.logical_and(distances < radius, cur_batch_id == torch_batch_ids)
        valid_pts = valid_pts and not mask.any()
        torch_batch_ids[cur_pt_iter] = cur_batch_id

    return valid_pts, end_time- start_time