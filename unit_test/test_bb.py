import os
import sys
import time
import numpy as np
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_bb(p_config_dict, p_data):
    
    # Get the parameters.
    batch_size = int(p_config_dict['batch_size'])
    num_pts = int(p_config_dict['num_pts'])
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

    # Use torch modules.
    device = torch.device("cuda:0")
    pc = pccnn_lib.pc.Pointcloud(selected_points, batch_ids, device=device)

    start_time = current_milli_time()
    bb = pccnn_lib.pc.BoundingBox(pc)
    end_time = current_milli_time()
    
    torch_min_pt = bb.min_.cpu().numpy()
    torch_max_pt = bb.max_.cpu().numpy()

    # Use numpy.
    selected_points = selected_points.reshape((-1, num_pts, num_dims))
    np_min_pt = np.amin(selected_points, 1)
    np_max_pt = np.amax(selected_points, 1)

    # Diff.
    equal_max = np.isclose(np_max_pt, torch_max_pt, rtol=1e-05, atol=1e-05)
    equal_min = np.isclose(np_min_pt, torch_min_pt, rtol=1e-05, atol=1e-05)

    return equal_max.all() and equal_min.all(), end_time - start_time