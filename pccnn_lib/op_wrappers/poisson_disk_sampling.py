import torch
import pccnn_lib_ops


class PoissonDiskSamlping(torch.autograd.Function):
    """Function to sample a point cloud using poisson disk sampling.
    """

    @staticmethod
    def forward(p_ctx, p_sorted_keys, p_neighbors, p_start_ids, p_grid_size):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_sorted_keys (tensor nxd): Point coordiante tensor.
            p_neighbors (tensor mx2): Neighbors tensor.
            p_start_ids (tensor n): Neighbors start indices tensor.
            p_grid_size (tensor d): Radii tensor.
        Returns:
            tensor k: Tensor with the selected ids.
        """
        result_tensor = pccnn_lib_ops.poisson_disk_sampling(p_sorted_keys, p_neighbors, 
            p_start_ids, p_grid_size)
        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """
        return None, None, None, None, None