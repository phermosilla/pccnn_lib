import math
import torch
import pccnn_lib
from pccnn_lib.op_wrappers import ComputePointConvBasis, ComputePointConv, ComputePointConvWeightVar
from .ILayer import ILayer, ILayerFactory
from torch_scatter import scatter_max

class PointConv(ILayer):
    """Class to implement a PointConv convolution.
    """

    def __init__(self, 
        p_dims, 
        p_in_features, 
        p_out_features, 
        p_hidden_size,
        p_constant_weight_var =True,
        p_const_var_value = 1.0,
        p_use_bn = False,
        p_use_gn = False,
        p_gn_size = 32):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
            p_hidden_size (int): Size of the hidden layer kernel.
            p_constant_weight_var (bool): Boolean that indicates
                if we use a weight initialization that guarantes
                the variance of the ouput always equal to 1.
            p_const_var_value (float): Constant used for the variance
                computations.
            p_use_bn (bool): True if we use batch norm.
            p_use_gn (bool): True if we use group norm.
            p_gn_size (int): Size of the groups in the group norm.
        """

        # Super class init.
        super(PointConv, self).__init__(
            p_dims, p_in_features, p_out_features,
            p_constant_weight_var,
            p_const_var_value)

        # Save parameters.
        self.use_bn_ = p_use_bn
        self.use_gn_ = p_use_gn
        self.gn_size_ = p_gn_size
        self.hidden_size_ = p_hidden_size
        self.init_proj_axis_ = False

        # Create the convolution parameters.
        self.proj_axis_ = torch.nn.Parameter(
            torch.empty(
                self.num_dims_,
                self.hidden_size_))
        self.proj_bias_ = torch.nn.Parameter(
            torch.empty(1, self.hidden_size_))
        self.conv_weights_ = torch.nn.Parameter(
            torch.empty(
                self.hidden_size_ * self.feat_input_size_,
                self.feat_output_size_))

        # Create the mlp used to transform the pdf.
        layer_list = []
        layer_list.append(torch.nn.Linear(1, 16))
        if self.use_bn_ or self.use_gn_:
            layer_list.append(torch.nn.BatchNorm1d(16))
        
        layer_list.append(torch.nn.ReLU())
        layer_list.append(torch.nn.Linear(16, 1))
        if self.use_bn_ or self.use_gn_:
            layer_list.append(torch.nn.BatchNorm1d(1))
        layer_list.append(torch.nn.Sigmoid())
        self.pdf_mlp_ = torch.nn.Sequential(*layer_list)

        # Batch norm of the basis projection.
        if self.use_bn_:
            self.basis_bn_ = torch.nn.BatchNorm1d(p_hidden_size)
        elif self.use_gn_:
            self.basis_gn_ = pccnn_lib.layers.norm_layers.GroupNormalization(p_hidden_size, self.gn_size_)

        # Reset parameters.
        self.reset_parameters()


    def reset_parameters(self):
        """Reset parameters.
        """
        stdv = math.sqrt(2.0 / (self.feat_input_size_*self.hidden_size_))
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
        
        # Compute the normalized pdf per neighborhood and apply the MLP transform.
        cur_pdf = torch.index_select(
            p_pc_in.pts_unnorm_pdf_, 0, 
            p_neighborhood.neighbors_[:,0].to(torch.int64))
        cur_pdf = torch.reciprocal(cur_pdf)
        max_pdf = scatter_max(cur_pdf, p_neighborhood.neighbors_[:,1].to(torch.int64), dim=0)[0]
        max_pdf = torch.index_select(
            max_pdf, 0, p_neighborhood.neighbors_[:,1].to(torch.int64))
        cur_pdf = cur_pdf/max_pdf
        cur_pdf = self.pdf_mlp_(torch.reshape(cur_pdf, (-1, 1)))

        cur_radius = torch.ones_like(p_radius)
        
        # Initialize the weights to have constant variance.
        if self.wamup_state_ and self.use_const_variance_init_:

            # Initialize the axis too.
            stdv = torch.sqrt(1.0 / (self.num_dims_ * p_radius[0] * p_radius[0]))
            self.proj_axis_.data.normal_(0.0, stdv)

             # Compute basis projection.
            basis_proj = ComputePointConvBasis.apply(p_pc_in.pts_, p_pc_out.pts_, 
                p_neighborhood.neighbors_, cur_radius, 
                self.proj_axis_, self.proj_bias_)
            if self.use_bn_:
                basis_proj = self.basis_bn_(basis_proj)
            elif self.use_gn_:
                self.basis_gn_(basis_proj, p_pc_out)
            basis_proj = torch.nn.ReLU()(basis_proj)*cur_pdf

            # Compute the optimal variance for the current batch.
            variance = ComputePointConvWeightVar.apply(
                basis_proj, p_in_features, 
                p_neighborhood.neighbors_, p_neighborhood.start_ids_)

            # Update the accumulated variance.
            self.__update_weight_var__(1.0/variance.item())

            # Initialize the weights.
            stdv = math.sqrt(self.out_constant_variance_/self.accum_weight_var_)
            self.conv_weights_.data.normal_(0.0, stdv)

        else:
            
            # Init projection axis.
            if self.init_proj_axis_:
                stdv = torch.sqrt(1.0 / (self.num_dims_ * p_radius[0] * p_radius[0]))
                self.proj_axis_.data.normal_(0.0, stdv)
                self.init_proj_axis_ = False

            # Compute basis projection.
            basis_proj = ComputePointConvBasis.apply(p_pc_in.pts_, p_pc_out.pts_, 
                p_neighborhood.neighbors_, cur_radius, 
                self.proj_axis_, self.proj_bias_)
            if self.use_bn_:
                basis_proj = self.basis_bn_(basis_proj)
            basis_proj = torch.nn.ReLU()(basis_proj)*cur_pdf

        # Compute convolution.
        conv_res = ComputePointConv.apply(
            basis_proj, p_in_features, p_neighborhood.neighbors_, 
            p_neighborhood.start_ids_, self.conv_weights_)

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



class PointConvFactory(ILayerFactory):
    """Interface of a layer actory.
    """

    def __init__(self, 
        p_num_basis,
        p_const_var_w_init = False,
        p_const_var_value = 1.0,
        p_use_bn = False,
        p_use_gn = False,
        p_gn_size = 32):
        """Constructor.
        """

        # Super class init.
        self.use_gn_ = p_use_gn
        self.use_bn_ = p_use_bn and not p_use_gn
        self.gn_size_ = p_gn_size
        super(PointConvFactory, self).__init__(
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
        Return ILayer object.
        """
        cur_conv = PointConv(p_dims, p_in_features, p_out_features, 
            self.num_basis_, self.conv_var_w_init_, self.const_var_value_,
            self.use_bn_, self.use_gn_, self.gn_size_)
        self.conv_list_.append(cur_conv)
        return cur_conv