import math
import torch
import pccnn_lib
from pccnn_lib.op_wrappers import ComputeMCConv, ComputeMCConvWeightVar
from .IConvLayer import IConvLayer, IConvLayerFactory

class MCConv(IConvLayer):
    """Class to implement a monte carlo convolution.
    """

    def __init__(self, 
        p_dims, 
        p_in_features, 
        p_out_features, 
        p_hidden_size,
        p_num_mlps = 1,
        p_constant_weight_var =True,
        p_const_var_value = 1.0):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
            p_hidden_size (int): Size of the hidden layer kernel.
            p_num_mlps (int): Number of different mlps.
            p_constant_weight_var (bool): Boolean that indicates
                if we use a weight initialization that guarantes
                the variance of the ouput always equal to 1.
            p_const_var_value (float): Constant used for the variance
                computations.
        """

        # Super class init.
        super(MCConv, self).__init__(
            p_dims, p_in_features, p_out_features,
            p_constant_weight_var,
            p_const_var_value)

        # Save parameters.
        self.hidden_size_ = p_hidden_size
        self.num_mlps_ = p_num_mlps

        # Create the convolution parameters.
        self.proj_axis_ = torch.nn.Parameter(
            torch.empty(
                self.num_dims_,
                self.hidden_size_*self.num_mlps_))
        self.proj_bias_ = torch.nn.Parameter(
            torch.empty(1, self.hidden_size_*self.num_mlps_))
        self.conv_weights_ = torch.nn.Parameter(
            torch.empty(
                self.num_mlps_, 
                self.hidden_size_ * self.feat_input_size_,
                self.feat_output_size_//p_num_mlps))

        # Reset parameters.
        self.reset_parameters()


    def reset_parameters(self):
        """Reset parameters.
        """
        stdv = math.sqrt(1.0 / (self.feat_input_size_*self.hidden_size_))
        self.conv_weights_.data.normal_(0.0, stdv)
        stdv = math.sqrt(1.0 / self.num_dims_)
        self.proj_axis_.data.normal_(0.0, stdv)
        self.proj_bias_.data.fill_(0.0)


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
        
        # Compute the unnormalized pdf by multiplying by the radius.
        mul_val = torch.prod(p_radius)
        cur_pdf = p_pc_in.pts_unnorm_pdf_ * mul_val
        
        # Initialize the weights to have constant variance.
        if self.wamup_state_ and self.use_const_variance_init_:

            # Compute the optimal variance for the current batch.
            variance = ComputeMCConvWeightVar.apply(
                p_pc_in.pts_, cur_pdf, p_in_features, p_pc_out.pts_,
                p_neighborhood.neighbors_, p_neighborhood.start_ids_, p_radius,
                self.proj_axis_, self.proj_bias_)*self.num_mlps_

            # Update the accumulated variance.
            self.__update_weight_var__(1.0/variance.item())

            # Initialize the weights.
            stdv = math.sqrt(self.out_constant_variance_/self.accum_weight_var_)
            self.conv_weights_.data.normal_(0.0, stdv)

        # Compute convolution.
        conv_res = ComputeMCConv.apply(
            p_pc_in.pts_, cur_pdf, p_in_features, p_pc_out.pts_,
            p_neighborhood.neighbors_, p_neighborhood.start_ids_, p_radius,
            self.proj_axis_, self.proj_bias_, self.conv_weights_)

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



class MCConvFactory(IConvLayerFactory):
    """Interface of a layer actory.
    """

    def __init__(self, 
        p_num_basis,
        p_const_var_w_init = False,
        p_num_mlps = 1,
        p_const_var_value = 1.0):
        """Constructor.
        """

        # Store the number of mlps.
        self.num_mlps_ = p_num_mlps

        # Super class init.
        super(MCConvFactory, self).__init__(
            p_num_basis,
            p_const_var_w_init,
            p_const_var_value)


    def create_conv_layer(self,
        p_dims, p_in_features, p_out_features):
        """Abstract mehod to create a layer.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
        Return IConvLayer object.
        """
        cur_conv = MCConv(p_dims, p_in_features, p_out_features, 
            self.num_basis_, self.num_mlps_, self.conv_var_w_init_, 
            self.const_var_value_)
        self.conv_list_.append(cur_conv)
        return cur_conv