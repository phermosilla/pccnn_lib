import torch
import pccnn_lib_ops


class ComputeMCConv(torch.autograd.Function):
    """Function to compute a Monte Carlo convolution.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_pt_pdf, p_features, p_samples, p_neighbors, 
        p_start_ids, p_radii, p_axis_proj, p_axis_bias, p_conv_weights):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordiante tensor.
            p_pt_pdf (tensor n): Point pdfs.
            p_features (tensor nxf): Point features.
            p_samples (tensor mxd): Sample coordinate tensor.
            p_neighbors (tensor tx2): Neighbors tensor.
            p_start_ids (tensor m): Neighbors start indices tensor.
            p_radii (tensor t): Radii tensor.
            p_axis_proj (tensor dxb): Axis for the projection.
            p_axis_bias (tensor 1xb): Bias for the projection.
            p_conv_weights (tensor b*fxof); Convolution weights.
        Returns:
            tensor n: Tensor with the convoluted features.
        """

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_pts, p_pt_pdf, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_axis_proj, p_axis_bias, p_conv_weights)

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.mc_conv(
            p_pts, p_pt_pdf, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_axis_proj, p_axis_bias, p_conv_weights)

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

            pts, pt_pdf, features, samples, neighbors, \
            start_ids, radii, axis_proj, axis_bias, \
                conv_weights = p_ctx.saved_tensors

            weight_grads, feat_grads, axis_grad, bias_grad, pdf_grad,\
                pt_grads, sample_grads = pccnn_lib_ops.mc_conv_grads(
                pts, pt_pdf, features, samples, neighbors, 
                start_ids, radii, axis_proj, axis_bias, 
                conv_weights, p_grads)

        else:
            weight_grads = None
            feat_grads = None
            axis_grad = None
            bias_grad = None
            pdf_grad = None
            pt_grads = None
            sample_grads = None

        return pt_grads, pdf_grad, feat_grads, sample_grads, None, None, \
            None, axis_grad, bias_grad, weight_grads


class ComputeMCConvWeightVar(torch.autograd.Function):
    """Function to compute the variance of the weights of a Monte Carlo
        convolution.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_pt_pdf, p_features, p_samples, p_neighbors, 
        p_start_ids, p_radii, p_axis_proj, p_axis_bias):
        """Forward.

        Args:
            p_ctx (context): Context.
            p_pts (tensor nxd): Point coordiante tensor.
            p_pt_pdf (tensor n): Point pdfs.
            p_features (tensor nxf): Point features.
            p_samples (tensor mxd): Sample coordinate tensor.
            p_neighbors (tensor tx2): Neighbors tensor.
            p_start_ids (tensor m): Neighbors start indices tensor.
            p_radii (tensor t): Radii tensor.
            p_axis_proj (tensor dxb): Axis for the projection.
            p_axis_bias (tensor 1xb): Bias for the projection.
        Returns:
            tensor n: Tensor with the pdf for each point.
        """

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.mc_conv_weight_var(
            p_pts, p_pt_pdf, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_axis_proj, p_axis_bias)

        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """


        return None, None, None, None, None, None, \
            None, None, None