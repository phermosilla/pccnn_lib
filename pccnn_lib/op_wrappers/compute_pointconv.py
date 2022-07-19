import torch
import pccnn_lib_ops

class ComputePointConvBasis(torch.autograd.Function):
    """Function to compute a PointConv convolution.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_samples, p_neighbors, 
        p_radii, p_axis_proj, p_axis_bias):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordiante tensor.
            p_samples (tensor mxd): Sample coordinate tensor.
            p_neighbors (tensor tx2): Neighbors tensor.
            p_radii (tensor t): Radii tensor.
            p_axis_proj (tensor dxb): Axis for the projection.
            p_axis_bias (tensor 1xb): Bias for the projection.
        Returns:
            tensor n: Tensor with the convoluted features.
        """

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_pts, p_samples, p_neighbors, 
            p_radii, p_axis_proj, p_axis_bias)

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.point_conv_basis(
            p_pts, p_radii, p_samples, p_neighbors, 
            p_axis_proj, p_axis_bias)

        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """

        pts, samples, neighbors, radii, axis_proj, axis_bias = p_ctx.saved_tensors

        pt_grads, sample_grads, axis_grad, bias_grad = pccnn_lib_ops.point_conv_basis_grads(
                pts, radii, samples, neighbors, 
                axis_proj, axis_bias, p_grads)

        return pt_grads, sample_grads, None, None, axis_grad, bias_grad


class ComputePointConv(torch.autograd.Function):
    """Function to compute a PointConv convolution.
    """

    @staticmethod
    def forward(p_ctx, p_basis_proj, p_features, p_neighbors, 
        p_start_ids, p_conv_weights):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_basis_proj (tensor nxb): Basis projection.
            p_features (tensor nxf): Point features.
            p_neighbors (tensor tx2): Neighbors tensor.
            p_start_ids (tensor m): Neighbors start indices tensor.
            p_conv_weights (tensor b*fxof); Convolution weights.
        Returns:
            tensor n: Tensor with the convoluted features.
        """

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_basis_proj, p_features, p_neighbors, p_start_ids, p_conv_weights)

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.point_conv(
            p_basis_proj, p_features, p_neighbors, p_start_ids, p_conv_weights)

        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """

        basis_proj, features, neighbors, start_ids, conv_weights = p_ctx.saved_tensors

        weights_grads, feat_grads, basis_grads = pccnn_lib_ops.point_conv_grads(
                basis_proj, features, neighbors, start_ids, conv_weights, p_grads)

        return basis_grads, feat_grads, None, None, weights_grads


class ComputePointConvWeightVar(torch.autograd.Function):
    """Function to compute the variance of the weights of a PointConv
        convolution.
    """

    @staticmethod
    def forward(p_ctx, p_basis, p_features, p_neighbors, p_start_ids):
        

        result_tensor = pccnn_lib_ops.point_conv_weight_var(
            p_basis, p_features, p_neighbors, p_start_ids)

        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """


        return None, None, None, None