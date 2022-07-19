import torch
import pccnn_lib_ops


class ComputeKPConv(torch.autograd.Function):
    """Function to compute a Monte Carlo convolution.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_features, p_samples, p_neighbors, 
        p_start_ids, p_radii, p_kernel_pts, p_sigma, p_conv_weights):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordiante tensor.
            p_features (tensor nxf): Point features.
            p_samples (tensor mxd): Sample coordinate tensor.
            p_neighbors (tensor tx2): Neighbors tensor.
            p_start_ids (tensor m): Neighbors start indices tensor.
            p_radii (tensor t): Radii tensor.
            p_kernel_pts (tensor dxb): Kernel points.
            p_sigma (float): Sigma.
            p_conv_weights (tensor b*fxof); Convolution weights.
        Returns:
            tensor n: Tensor with the pdf for each point.
        """

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_pts, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_kernel_pts, p_conv_weights)
        p_ctx.sigma_ = p_sigma

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.kp_conv(
            p_pts, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_kernel_pts, p_sigma, p_conv_weights)

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

            pts, features, samples, neighbors, \
            start_ids, radii, kernel_pts, conv_weights = p_ctx.saved_tensors
            sigma = p_ctx.sigma_

            weight_grads, feat_grads, kernel_pts_grads, \
                pt_grads, sample_grads = pccnn_lib_ops.kp_conv_grads(
                pts, features, samples, neighbors, 
                start_ids, radii, kernel_pts, sigma, 
                conv_weights, p_grads)

        else:
            weight_grads = None
            feat_grads = None
            kernel_pts_grads = None
            pt_grads = None
            sample_grads = None

        return pt_grads, feat_grads, sample_grads, None, None, \
            None, kernel_pts_grads, None, weight_grads


class ComputeKPConvWeightVar(torch.autograd.Function):
    """Function to compute the variance of the weights of a KPConv.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_features, p_samples, p_neighbors, 
        p_start_ids, p_radii, p_kernel_pts, p_sigma):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordiante tensor.
            p_features (tensor nxf): Point features.
            p_samples (tensor mxd): Sample coordinate tensor.
            p_neighbors (tensor tx2): Neighbors tensor.
            p_start_ids (tensor m): Neighbors start indices tensor.
            p_radii (tensor t): Radii tensor.
            p_kernel_pts (tensor dxb): Kernel points.
            p_sigma (float): Sigma.
        Returns:
            tensor n: Tensor with the pdf for each point.
        """

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.kp_conv_weight_var(
            p_pts, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_kernel_pts, p_sigma)

        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """


        return None, None, None, None, None, \
            None, None, None