import torch
import pccnn_lib_ops


class ComputeSPHConv(torch.autograd.Function):
    """Function to compute a spherical convolution.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_features, p_samples, p_neighbors, 
        p_start_ids, p_radii, p_radial_bins, p_azimuz_bins, 
        p_polar_bins, p_conv_weights):
        

        # Save for backwards if gradients for points are required.
        p_ctx.save_for_backward(
            p_pts, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_radial_bins, p_azimuz_bins, 
            p_polar_bins, p_conv_weights)

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.sph_conv(
            p_pts, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_radial_bins, p_azimuz_bins, 
            p_polar_bins, p_conv_weights)

        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """


        pts, features, samples, neighbors, \
            start_ids, radii, radial_bins, \
            azimuz_bins, polar_bins, conv_weights = p_ctx.saved_tensors

        weight_grads, feat_grads = pccnn_lib_ops.sph_conv_grads(
            pts, features, samples, neighbors, 
            start_ids, radii, radial_bins, azimuz_bins, 
            polar_bins, conv_weights, p_grads)


        return None, feat_grads, None, None, None, \
            None, None, None, None, weight_grads


class ComputeSPHConvWeightVar(torch.autograd.Function):
    """Function to compute the variance of the weights of a SPHConv.
    """

    @staticmethod
    def forward(p_ctx, p_pts, p_features, p_samples, p_neighbors, 
        p_start_ids, p_radii, p_radial_bins, p_azimuz_bins, 
        p_polar_bins):

        # Compute the pdf.
        result_tensor = pccnn_lib_ops.sph_conv_weight_var(
            p_pts, p_features, p_samples, p_neighbors, 
            p_start_ids, p_radii, p_radial_bins, p_azimuz_bins, 
            p_polar_bins)

        return result_tensor


    @staticmethod
    def backward(p_ctx, p_grads):
        """Backward.

        Args:
            p_ctx (context): Context.
            p_grads (tensor bxwxh): Input gradients.
        """


        return None, None, None, None, None, \
            None, None, None, None