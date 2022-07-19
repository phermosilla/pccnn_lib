import math
import time
import numpy as np
import torch
import pccnn_lib
from pccnn_lib.op_wrappers import ComputePCCNN, ComputePCCNNWeightVar
from pccnn_lib.pc.layers.kernel_points.kernel_points import kernel_pts_dict
from pccnn_lib.py_utils.pc_utils import rotate_pc_3d
from .IConvLayer import IConvLayer, IConvLayerFactory

class PCCNN(IConvLayer):
    """Class to implement a PCCNN.
    """

    def __init__(self, 
        p_dims, 
        p_in_features, 
        p_out_features, 
        p_num_kernel_pts,
        p_constant_weight_var =True,
        p_const_var_value = 1.0):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
            p_num_kernel_pts (int): Number of kernel points.
            p_constant_weight_var (bool): Boolean that indicates
                if we use a weight initialization that guarantes
                the variance of the ouput always equal to 1.
            p_const_var_value (float): Constant used for the variance
                computations.
        """

        # Super class init.
        super(PCCNN, self).__init__(
            p_dims, p_in_features, p_out_features,
            p_constant_weight_var,
            p_const_var_value)

        # Save parameters.
        self.num_kernel_pts_ = p_num_kernel_pts

        # Load kernel points.
        np_kernel_pts = kernel_pts_dict[str(p_dims)+"_"+str(self.num_kernel_pts_)].astype(np.float32)
        distances = np.linalg.norm(np_kernel_pts, axis=-1).astype(np.float32)
        max_dist = np.amax(distances)
        distances = distances / max_dist
        np_kernel_pts = np_kernel_pts / max_dist
        min_distance = np.amin(distances[1:])
        if np_kernel_pts.shape[-1] == 3:
            np_kernel_pts, _ = rotate_pc_3d(np.random.RandomState(int(time.time())), np_kernel_pts)
        np_kernel_pts = np_kernel_pts.astype(np.float32)

        self.register_buffer('kernel_pts_', torch.from_numpy(np_kernel_pts/(1.0+min_distance)))
        self.register_buffer('sigma_', torch.tensor((min_distance*0.4)/(1.0+min_distance)))
        self.sigma_cpu_ = None

        # Create the convolution parameters.
        self.conv_weights_ = torch.nn.Parameter(
            torch.empty(
                self.num_kernel_pts_ * self.feat_input_size_,
                self.feat_output_size_))

        # Reset parameters.
        self.reset_parameters()


    def reset_parameters(self):
        """Reset parameters.
        """
        stdv = math.sqrt(1.0 / (self.feat_input_size_*self.num_kernel_pts_))
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
        
        # Compute the unnormalized pdf by multiplying by the radius.
        mul_val = torch.prod(p_radius)
        cur_pdf = p_pc_in.pts_unnorm_pdf_ * mul_val

        # Get sigma value to cpu.
        if self.sigma_cpu_ is None:
            self.sigma_cpu_ = self.sigma_.item()

        # Initialize the weights to have constant variance.
        if self.wamup_state_ and self.use_const_variance_init_:

            # Compute the optimal variance for the current batch.
            variance = ComputePCCNNWeightVar.apply(
                p_pc_in.pts_, cur_pdf, p_in_features, p_pc_out.pts_,
                p_neighborhood.neighbors_, p_neighborhood.start_ids_, p_radius,
                self.kernel_pts_, self.sigma_cpu_)
    
            # Update the accumulated variance.
            self.__update_weight_var__(1.0/variance.item())

            # Initialize the weights.
            stdv = math.sqrt(self.out_constant_variance_/self.accum_weight_var_)
            self.conv_weights_.data.normal_(0.0, stdv)

        # Compute convolution.
        conv_res = ComputePCCNN.apply(
            p_pc_in.pts_, cur_pdf, p_in_features, p_pc_out.pts_,
            p_neighborhood.neighbors_, p_neighborhood.start_ids_, p_radius,
            self.kernel_pts_, self.sigma_cpu_, self.conv_weights_)

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



class PCCNNFactory(IConvLayerFactory):
    """Interface of a layer actory.
    """

    def __init__(self, 
        p_num_kernel_pts,
        p_const_var_w_init = False,
        p_const_var_value = 1.0):
        """Constructor.
        """

        # Super class init.
        super(PCCNNFactory, self).__init__(
            p_num_kernel_pts,
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
        cur_conv = PCCNN(p_dims, p_in_features, p_out_features, 
            self.num_basis_, self.conv_var_w_init_, 
            self.const_var_value_)
        self.conv_list_.append(cur_conv)
        return cur_conv
