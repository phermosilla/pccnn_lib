import torch
import pccnn_lib_ops


class ComputePDF(torch.autograd.Function):
    """Function to compute the pdf for a set of points.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_neighbors, p_start_ids, 
        p_bandwidth, p_manifold_dims):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordiante tensor.
            p_neighbors (tensor mx2): Neighbors tensor.
            p_start_ids (tensor n): Neighbors start indices tensor.
            p_bandwidth (tensor d): Badnwidth tensor.
            p_manifold_dims (int): Dimension of the data manifold.
        Returns:
            tensor n: Tensor with the pdf for each point.
        """
        # Save for backwards if gradients for points are required.
        if p_pts.requires_grad:
            p_ctx.save_for_backward(p_pts, p_neighbors, p_start_ids, 
                p_bandwidth)
            p_ctx.manifold_dims_ = p_manifold_dims

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.compute_pdf(
            p_manifold_dims, p_bandwidth, 
            p_pts, p_neighbors, p_start_ids)

        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """

        # If required compute gradients.
        if len(p_ctx.saved_tensors) > 0:
            minfold_dims = p_ctx.manifold_dims_
            pts, neighbors, start_ids, bandwidth = p_ctx.saved_tensors
            point_grads = pccnn_lib_ops.compute_pdf_grads(
                minfold_dims, bandwidth,
                pts, neighbors, start_ids, p_grads)
        else:
            point_grads = None

        return point_grads, None, None, None, None