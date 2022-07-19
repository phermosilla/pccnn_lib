import torch
import pccnn_lib_ops


class ComputeKeys(torch.autograd.Function):
    """Function to compute the grid keys for a set of points.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_batch_ids, p_aabb_min, p_grid_size, p_cell_size):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordiante tensor.
            p_batch_ids (tensor n): Batch ids tensor.
            p_aabb_min (tensor bxd): AABB min point tensor.
            p_grid_size (tensor d): Grid size tensor.
            p_cell_size (tensor d): Cell size tensor.
        Returns:
            tensor n: Tensor with the keys for each point.
        """
        result_tensor = pccnn_lib_ops.compute_keys(p_pts, p_batch_ids, p_aabb_min,
            p_grid_size, p_cell_size)
        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """
        return None, None, None, None, None