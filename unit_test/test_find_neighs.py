import os
import sys
import time
import numpy as np
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_find_neighs(p_config_dict, p_data):
    
    # Get the parameters.
    batch_size = int(p_config_dict['batch_size'])
    num_pts = int(p_config_dict['num_pts'])
    radius = float(p_config_dict['radius'])
    num_dims = int(p_config_dict['num_dims'])
    max_num_neighs = int(p_config_dict['max_num_neighs'])

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
    neigh = pccnn_lib.pc.Neighborhood(grid, pc, max_num_neighs)

    end_time = current_milli_time()
    torch_neighbors = neigh.neighbors_.to(torch.int32).cpu().numpy()
    torch_start_ids = neigh.start_ids_.to(torch.int32).cpu().numpy()

    # Use numpy.
    np_neighbors = []
    np_start_ids = []
    for cur_pt_iter, cur_pt in enumerate(selected_points):
        cur_batch_id = batch_ids[cur_pt_iter]
        diff_pts = selected_points-cur_pt.reshape((1, num_dims))
        distances = np.sqrt(np.sum(diff_pts*diff_pts, axis=-1))
        mask_neighbors = np.logical_and(distances < radius, batch_ids == cur_batch_id)
        neigh_ids = np.where(mask_neighbors)[0]
        for cur_neigh in neigh_ids:
            np_neighbors.append([cur_neigh, cur_pt_iter])
        np_start_ids.append(len(np_neighbors))
    np_neighbors = np.array(np_neighbors)
    np_start_ids = np.array(np_start_ids)
        
    # Diff.
    equal_neighs = True
    for cur_pt_iter in range(len(np_start_ids)):
        cur_torch_start_id = 0
        cur_np_start_id = 0
        if cur_pt_iter > 0:
            cur_torch_start_id = torch_start_ids[cur_pt_iter-1]
            cur_np_start_id = np_start_ids[cur_pt_iter-1]
        cur_torch_end_id = torch_start_ids[cur_pt_iter]
        cur_np_end_id = np_start_ids[cur_pt_iter]
        num_pts_np = cur_np_end_id-cur_np_start_id
        num_pts_torch = cur_torch_end_id-cur_torch_start_id
        if (num_pts_np == num_pts_torch) or \
            (max_num_neighs > 0 and \
            num_pts_np > max_num_neighs and \
            num_pts_torch == max_num_neighs):
            found_all_neighs = True
            for cur_torch_neigh in torch_neighbors[cur_torch_start_id:cur_torch_end_id]:
                found_neigh = False
                for cur_np_neigh in np_neighbors[cur_np_start_id:cur_np_end_id]:
                    if cur_np_neigh[0] == cur_torch_neigh[0]:
                        found_neigh = True
                found_all_neighs = found_all_neighs and found_neigh
            equal_neighs = equal_neighs and found_all_neighs
        else:
            equal_neighs = False

    return equal_neighs, end_time- start_time