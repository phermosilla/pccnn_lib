import os
import sys
import time
import numpy as np
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_grid_ds(p_config_dict, p_data):
    
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
    grid.build_ds()

    end_time = current_milli_time()
    torch_grid_ds = grid.grid_ds_.to(torch.int32).cpu().numpy()

    # Use numpy.
    # Compute grid cell ids.
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
    
    # Sort points.
    sorted_indices = np.argsort(np_cell_ids)
    sorted_cell_ids = np_cell_ids[sorted_indices]

    # Build grid.
    np_grid_ds = np.full((batch_size, grid_size[0], grid_size[1], 2), 
        0, dtype=np.int32)
    batch_mul_factor = accum_mul
    dim_1_mul_factor = batch_mul_factor//grid_size[0]
    dim_2_mul_factor = dim_1_mul_factor//grid_size[1]
    for cur_iter, cur_cell_id in enumerate(sorted_cell_ids):
        cur_batch_id = cur_cell_id//batch_mul_factor
        cur_dim_1 = (cur_cell_id//dim_1_mul_factor)%grid_size[0]
        cur_dim_2 = (cur_cell_id//dim_2_mul_factor)%grid_size[1]
        cur_modif_cell_id = cur_cell_id//dim_2_mul_factor
        if cur_iter > 0:
            prev_cell_id = sorted_cell_ids[cur_iter-1]
            prev_batch_id = prev_cell_id//batch_mul_factor
            prev_dim_1 = (prev_cell_id//dim_1_mul_factor)%grid_size[0]
            prev_dim_2 = (prev_cell_id//dim_2_mul_factor)%grid_size[1]
            prev_modif_cell_id = prev_cell_id//dim_2_mul_factor
            if prev_modif_cell_id != cur_modif_cell_id:
                np_grid_ds[cur_batch_id, cur_dim_1, cur_dim_2, 0] = cur_iter
                np_grid_ds[prev_batch_id, prev_dim_1, prev_dim_2, 1] = cur_iter

    last_cell_id = sorted_cell_ids[-1]
    last_batch_id = last_cell_id//batch_mul_factor
    last_dim_1 = (last_cell_id//dim_1_mul_factor)%grid_size[0]
    last_dim_2 = (last_cell_id//dim_2_mul_factor)%grid_size[1]
    np_grid_ds[last_batch_id, last_dim_1, last_dim_2, 1] = len(sorted_cell_ids)

    # Diff.
    equal_array = np.equal(torch_grid_ds, np_grid_ds)

    return equal_array.all(), end_time - start_time