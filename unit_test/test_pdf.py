import os
import sys
import time
import numpy as np
from sklearn.neighbors import KernelDensity
import torch

current_milli_time = lambda: time.time() * 1000.0

import pccnn_lib

def test_pdf(p_config_dict, p_data):
    
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

    # Use torch modules.
    device = torch.device("cuda:0")
    pc = pccnn_lib.pc.Pointcloud(selected_points, batch_ids, device=device)
    start_time = current_milli_time()

    pc.compute_pdf(badnwidth)

    end_time = current_milli_time()
    torch_pdf = pc.pts_pdf_.to(torch.float32).cpu().numpy()

    # Nupmy.
    selected_points = selected_points.reshape((batch_size, num_pts, num_dims))
    np_pdf = []
    for cur_batch_id in range(batch_size):
        kdeSkl = KernelDensity(bandwidth=radius)
        kdeSkl.fit(selected_points[cur_batch_id, :, :])
        logPdf = kdeSkl.score_samples(selected_points[cur_batch_id, :, :])
        np_pdf.append(np.exp(logPdf))
    np_pdf = np.concatenate(np_pdf, axis=0)

    # Diff.
    equal = np.isclose(np_pdf, torch_pdf, rtol=1e-02, atol=1e-02)

    return equal.all(), end_time- start_time