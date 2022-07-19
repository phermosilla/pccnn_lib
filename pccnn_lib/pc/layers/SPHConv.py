import math
import time
import numpy as np
import torch
import pccnn_lib
from pccnn_lib.op_wrappers import ComputeSPHConv, ComputeSPHConvWeightVar
from .ILayer import ILayer, ILayerFactory

class SPHConv(ILayer):
    """Class to implement a SPHConv.
    """

    def __init__(self, 
        p_dims, 
        p_in_features, 
        p_out_features, 
        p_num_radial_bins = 1,
        p_num_azimuz_bins = 5,
        p_num_polar_bins = 3,
        p_constant_weight_var =True,
        p_const_var_value = 1.0):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
            p_num_radial_bins (int): Number of bins in the radial axis.
            p_num_azimuz_bins (int): Number of bins on the azimuz angle.
            p_num_polar_bins (int): Number of bins on the polar angle.
            p_constant_weight_var (bool): Boolean that indicates
                if we use a weight initialization that guarantes
                the variance of the ouput always equal to 1.
            p_const_var_value (float): Constant used for the variance
                computations.
        """

        # Super class init.
        super(SPHConv, self).__init__(
            p_dims, p_in_features, p_out_features,
            p_constant_weight_var,
            p_const_var_value)

        # Save parameters.
        self.num_radial_bins_ = p_num_radial_bins
        self.num_azimuz_bins_ = p_num_azimuz_bins
        self.num_polar_bins_ = p_num_polar_bins
        self.num_basis_ = p_num_radial_bins * p_num_azimuz_bins * p_num_polar_bins + 1

        # Compute bins.
        radial_bins = []
        radial_incr = 1.0 / float(p_num_radial_bins)
        for i in range(p_num_radial_bins):
            radial_bins.append(radial_incr*(i+1))
        azimuz_bins = []
        azimuz_incr = (2.0*math.pi) / float(p_num_azimuz_bins)
        for i in range(p_num_azimuz_bins):
            azimuz_bins.append(azimuz_incr*(i+1))
        polar_bins = []
        polar_incr = (math.pi) / float(p_num_polar_bins)
        for i in range(p_num_polar_bins):
            polar_bins.append(polar_incr*(i+1))

        self.register_buffer('radial_bins_', torch.from_numpy(np.array(radial_bins).astype(np.float32)))
        self.register_buffer('azimuz_bins_', torch.from_numpy(np.array(azimuz_bins).astype(np.float32)))
        self.register_buffer('polar_bins_', torch.from_numpy(np.array(polar_bins).astype(np.float32)))
        self.sigma_cpu_ = None

        # Create the convolution parameters.
        self.conv_weights_ = torch.nn.Parameter(
            torch.empty(
                self.num_basis_ * self.feat_input_size_,
                self.feat_output_size_))

        # Reset parameters.
        self.reset_parameters()


    def reset_parameters(self):
        """Reset parameters.
        """
        stdv = math.sqrt(1.0 / (self.feat_input_size_*self.num_basis_))
        self.conv_weights_.data.normal_(0.0, stdv)


    def __compute_convolution__(self,
        p_pc_in,
        p_pc_out,
        p_in_features,
        p_radius,
        p_neighborhood):
        """Abstract mehod to implement a convolution.

        Args:
            p_pc_in (Pointcloud): Input point cloud.
            p_pc_out (Pointcloud): Output point cloud.
            p_in_features (tensor nxfi): Input features.
            p_radius (float): Convolution radius.
            p_neighborhood (Neighborhood): Input neighborhood. If
                None, a new neighborhood is computed.
            p_max_neighs (int): Max number of neighbors.

        Returns:
            tensor mxfo: Output features.
        """

        # Initialize the weights to have constant variance.
        if self.wamup_state_ and self.use_const_variance_init_:

            # Compute the optimal variance for the current batch.
            variance = ComputeSPHConvWeightVar.apply(
                p_pc_in.pts_, p_in_features, p_pc_out.pts_,
                p_neighborhood.neighbors_, p_neighborhood.start_ids_, p_radius,
                self.radial_bins_, self.azimuz_bins_, self.polar_bins_)
    
            # Update the accumulated variance.
            self.__update_weight_var__(1.0/variance.item())

            # Initialize the weights.
            stdv = math.sqrt(self.out_constant_variance_/self.accum_weight_var_)
            self.conv_weights_.data.normal_(0.0, stdv)

        # Compute convolution.
        conv_res = ComputeSPHConv.apply(
            p_pc_in.pts_, p_in_features, p_pc_out.pts_,
            p_neighborhood.neighbors_, p_neighborhood.start_ids_, p_radius,
            self.radial_bins_, self.azimuz_bins_, self.polar_bins_,
            self.conv_weights_)

        return conv_res


    def __create_neighborhood__(self,
        p_pc_in,
        p_pc_out,
        p_radius,
        p_max_neighs):
        """Method to create a neighborhood object.

        Args:
            p_pc_in (Pointcloud): Input point cloud.
            p_pc_out (Pointcloud): Output point cloud.
            p_radius (float): Convolution radius.
            p_max_neighs (int): Max number of neighbors. 
        Return 
            Neighborhood: Output neighborhood.
        """
        bb = pccnn_lib.pc.BoundingBox(p_pc_in)
        grid = pccnn_lib.pc.Grid(p_pc_in, bb, p_radius)
        grid.compute_cell_ids()
        grid.build_ds()
        return pccnn_lib.pc.Neighborhood(grid, p_pc_out, p_max_neighs)



class SPHConvFactory(ILayerFactory):
    """Interface of a layer actory.
    """

    def __init__(self, 
        p_num_radial_bins,
        p_num_azimuz_bins,
        p_num_polar_bins,
        p_const_var_w_init = False,
        p_const_var_value = 1.0):
        """Constructor.
        """

        # Save parameters.
        self.radial_bins_ = p_num_radial_bins
        self.azimuz_bins_ = p_num_azimuz_bins
        self.polar_bins_ = p_num_polar_bins

        # Super class init.
        super(SPHConvFactory, self).__init__(
            p_num_radial_bins*p_num_azimuz_bins*p_num_polar_bins + 1,
            p_const_var_w_init,
            p_const_var_value)


    def create_conv_layer(self,
        p_dims, p_in_features, p_out_features):
        """Abstract mehod to create a layer.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
        Return ILayer object.
        """
        cur_conv = SPHConv(
            p_dims, p_in_features, p_out_features, 
            self.radial_bins_, self.azimuz_bins_,
            self.polar_bins_, self.conv_var_w_init_, 
            self.const_var_value_)
        self.conv_list_.append(cur_conv)
        return cur_conv