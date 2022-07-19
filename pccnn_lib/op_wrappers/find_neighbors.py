import torch
import pccnn_lib_ops


class FindNeighbors(torch.autograd.Function):
    """Function to the neighbors of a point on a regular grid.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_pt_keys, p_samples, p_sample_keys, 
            p_grid_ds, p_grid_size, p_radius, p_max_neighbors):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordiante tensor.
            p_pt_keys (tensor n): Point grid keys.
            p_samples (tensor mxd): Sample coordiante tensor.
            p_sample_keys (tensor m): Sample grid keys.
            p_grid_ds (tensor bxwxhx2): Grid data structure.
            p_grid_size (tensor d): Grid size tensor.
            p_radius (tensor d): Radius of the query.
            p_max_neighbors (int): Maximum number of neighbors.
        Returns:
            tensor kx2: Tensor with the list of neighbors.
            tensor m: Tensor with the start indices for the samples.
        """
        result_neighbors, result_start_ids = pccnn_lib_ops.find_neighbors(
            p_pts, p_pt_keys, p_samples, p_sample_keys,
            p_grid_ds, p_grid_size, p_radius, p_max_neighbors)
        return result_neighbors, result_start_ids


    @staticmethod
    def backward(p_ctx, p_grads_neigh, p_grads_start_ids):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads_neigh (tensor bxwxh): Input gradients.
            p_grads_start_ids (tensor bxwxh): Input gradients.
        """
        return None, None, None, None, None, None