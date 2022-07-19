import torch
import pccnn_lib_ops


class BuildGridDS(torch.autograd.Function):
    """Function to build the grid data structure.
    """

    @staticmethod
    def forward(p_ctx, p_keys, p_grid_size, p_batch_size):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_keys (tensor n): Keys per point.
            p_grid_size (tensor d): Number of cells per dimension.
            p_batch_size (tensor): Batch size.
        Returns:
            tensor bxwxh: Data structure to fast access the points.
        """
        final_shape = [p_batch_size.tolist()] + p_grid_size[0:2].tolist() + [2]
        result_tensor = pccnn_lib_ops.build_grid_ds(p_keys, p_grid_size, final_shape)
        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """
        return None, None, None