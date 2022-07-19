from abc import ABC, abstractmethod
import math
import torch
import pccnn_lib


class IConvLayer(torch.nn.Module, ABC):
    """Interface of layer for a point convolution.
    """

    def __init__(self, 
        p_dims, 
        p_in_features, 
        p_out_features,
        p_constant_weight_var =True,
        p_const_var_value = 1.0):
        """Constructor.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
            p_constant_weight_var (bool): Boolean that indicates
                if we use a weight initialization that guarantes
                the variance of the ouput always equal to a constant.
            p_const_var_value (float): Constant used for the variance
                computations.
        """

        # Super class init.
        super(IConvLayer, self).__init__()

        # Save params.
        self.num_dims_ = p_dims
        self.feat_input_size_ = p_in_features
        self.feat_output_size_ = p_out_features
        self.wamup_state_ = False
        self.use_const_variance_init_ = p_constant_weight_var
        self.out_constant_variance_ = p_const_var_value
        self.accum_weight_var_ = 0.0
        self.accum_weight_var_counter_ = 1.0


    def set_init_warmup_state(self, p_init_warmup_state):
        """Method to define the warmup state of the layer. If true
        and const variance init is used, the layer accumulate statistics
        of the optimal variance for the batches processed.

        Args:
            p_init_warmup_state (bool): Boolean that indicates if the
                layer is in warmup state.
        """
        self.wamup_state_ = p_init_warmup_state


    def __update_weight_var__(self, p_new_var):
        """Method to update the accumulated variance.

        Args:
            p_new_var (float): New variance to accumulate.
        """
        self.accum_weight_var_ += (p_new_var - self.accum_weight_var_)/self.accum_weight_var_counter_
        self.accum_weight_var_counter_ += 1.0


    @abstractmethod
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
        pass

    
    @abstractmethod
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
        pass


    def forward(self, 
        p_pc_in,
        p_pc_out,
        p_in_features,
        p_radius,
        p_neighborhood = None,
        p_max_neighs = 0):
        """Forward pass.

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
            Neighborhood: Output neighborhood.
        """

        # Create the radius tensor.
        if not(torch.is_tensor(p_radius)):
            radius = torch.as_tensor(p_radius, device=p_pc_in.pts_.device)
        else:
            radius = p_radius

        # If neighborhood is None, compute it.
        if p_neighborhood is None:
            neighborhood = self.__create_neighborhood__(
                p_pc_in, p_pc_out, radius, p_max_neighs)
        else:
            neighborhood = p_neighborhood        

        # Compute the convolution.
        out_features = self.__compute_convolution__(
            p_pc_in,
            p_pc_out,
            p_in_features,
            radius,
            neighborhood)

        # Return results.
        if p_neighborhood is None:
            return out_features, neighborhood
        else:
            return out_features


class IConvLayerFactory(ABC):
    """Interface of a layer actory.
    """

    def __init__(self,
        p_num_basis,
        p_const_var_w_init,
        p_const_var_value = 1.0):
        """Constructor.
        """

        # Super class init.
        super(IConvLayerFactory, self).__init__()

        # Save parameters.
        self.num_basis_ = p_num_basis
        self.conv_var_w_init_ = p_const_var_w_init
        self.const_var_value_ = p_const_var_value

        # Initialize the convolution list.
        self.conv_list_ = []


    def set_init_warmup_state_convs(self, p_state):
        """Method to define the warmup state of the layer. If true
        and const variance init is used, the layer accumulate statistics
        of the optimal variance for the batches processed.

        Args:
            p_state (bool): Boolean that indicates if the
                layer is in warmup state.
        """
        for cur_conv in self.conv_list_:
            cur_conv.set_init_warmup_state(p_state)

    
    def define_num_basis(self, p_num_basis):
        """Method to define the number of basis of the convolution.

        Args:
            p_num_basis (int): Number of basis.
        """
        self.num_basis_ = p_num_basis


    @abstractmethod
    def create_conv_layer(self,
        p_dims, p_in_features, p_out_features):
        """Abstract mehod to create a layer.

        Args:
            p_dims (int): Number of dimensions.
            p_in_features (int): Number of input features.
            p_out_features (int): Number of output features.
        Return IConvLayer object.
        """
        pass