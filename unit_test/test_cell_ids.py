import os
import sys
import time
import numpy as np
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_cell_ids(p_config_dict, p_data):
    
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

    bb = pccnn_lib.pc.BoundingBox(pc)
    grid = pccnn_lib.pc.Grid(pc, bb, cell_size)
    grid.compute_cell_ids()

    end_time = current_milli_time()
    torch_cell_ids = grid.cell_ids_.to(torch.int32).cpu().numpy()

    # Use numpy.
    selected_points = selected_points.reshape((-1, num_pts, num_dims))
    grid_size = grid.num_cells_.cpu().numpy()
    diff_pts = selected_points - np.amin(selected_points, 1, keepdims=True)
    np_cell_indexs = np.floor(diff_pts/cell_size.reshape((1, 1, num_dims)))
    np_cell_indexs = np_cell_indexs.astype(np.int32).reshape((-1, num_dims))
    np_cell_indexs = np.maximum(np.zeros((1, num_dims), dtype=np.int32), 
        np.minimum(np_cell_indexs, grid_size.reshape((1, num_dims))))

    np_cell_ids = np_cell_indexs[:, -1]
    accum_mul = grid_size[-1]
    for cur_dim in range(num_dims-2, -1, -1):
        np_cell_ids = np_cell_ids + \
            np_cell_indexs[:, cur_dim]*accum_mul
        accum_mul = accum_mul * grid_size[cur_dim]
    np_cell_ids = np_cell_ids + batch_ids*accum_mul
    
    # Diff.
    equal_array = np.equal(torch_cell_ids, np_cell_ids)

    return equal_array.all(), end_time - start_time