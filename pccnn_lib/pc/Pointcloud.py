import numpy as np
import torch
from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_add

from pccnn_lib.pc.BoundingBox import BoundingBox
from pccnn_lib.pc.Grid import Grid
from pccnn_lib.pc.Neighborhood import Neighborhood
from pccnn_lib.op_wrappers import ComputePDF

class Pointcloud(object):
    """Class to represent a point cloud.
    """

    def __init__(self, p_pts, p_batch_ids, 
        p_manifold_dims = None, **kwargs):
        '''Constructor.
        
        Args:
            p_pts (np.array nxd): Point coordinates.
            p_batch_ids (np.array n): Point batch ids.
            p_manifold_dims (int): Number of dimension of the data manifold.
                A plane in 3D space will have 2 dimension whilst a line
                will have  only one. If None the dimensions of the data
                is used.
        '''

        # Save data manifold.
        if p_manifold_dims is None:
            self.dim_manifold_ = p_pts.shape[1]
        else:
            self.dim_manifold_ = p_manifold_dims

        # Check if requires_grad is specified.
        self.pts_with_grads_ = False
        if 'requires_grad' in kwargs:
            self.pts_with_grads_ = kwargs['requires_grad']
            del kwargs['requires_grad']

        # Create the tensors.
        self.pts_ = torch.as_tensor(p_pts, **kwargs)
        self.batch_ids_ = torch.as_tensor(p_batch_ids, **kwargs)
        self.batch_size_ = torch.max(self.batch_ids_) + 1

        # Update requires_grad if needed.
        if self.pts_with_grads_:
            self.pts_.requires_grad = True

        # Initialize pdf.
        self.pts_pdf_ = None
        self.pts_unnorm_pdf_ = None


    def to_device(self, p_device):
        """Method to move the tensors to a specific device.

        Return:
            device p_device: Destination device.
        """
        self.pts_ = self.pts_.to(p_device)
        self.batch_ids_ = self.batch_ids_.to(p_device)
        self.batch_size_ = self.batch_size_.to(p_device)
        if not(self.pts_unnorm_pdf_ is None):
            self.pts_unnorm_pdf_ = self.pts_unnorm_pdf_.to(p_device)
            self.pts_pdf_ = self.pts_pdf_.to(p_device)

    
    def get_num_points_per_batch(self):
        """Method to get the number of points per batch.

        Return:
            int tensor: Number of points per each batch.
        """
        with torch.no_grad():
            aux_ones = torch.ones_like(self.batch_ids_)
            num_pts = torch.zeros((self.batch_size_))\
                .to(torch.int32).to(self.batch_ids_.device)
            num_pts.index_add_(0, self.batch_ids_.to(torch.int64), aux_ones)
        return num_pts


    def compute_pdf(self, p_badnwidth):
        """Method to compute the pdf of the point cloud.

        Args:
            p_badnwidth (tensor d): Bandwidth used for the computations.
        """

        # Define the neighborhood radius for efficient computations.
        bandwidth = torch.as_tensor(p_badnwidth, device=self.pts_.device)
        radii = bandwidth * 4.0

        # Create the neighborhood.
        aabb = BoundingBox(self)
        grid = Grid(self, aabb, radii)
        grid.compute_cell_ids()
        grid.build_ds()
        neigh = Neighborhood(grid, self, 300)

        # Compute the pdfs.
        self.pts_unnorm_pdf_ = ComputePDF.apply(self.pts_, neigh.neighbors_, 
            neigh.start_ids_, bandwidth, self.dim_manifold_)

        # Get the number of points per batch.
        num_pts_x_batch = self.get_num_points_per_batch()
        num_pts = torch.index_select(num_pts_x_batch, 0, 
                            self.batch_ids_.to(torch.int64))

        # Normalize to the number of points.
        self.pts_pdf_ = self.pts_unnorm_pdf_ / num_pts.to(torch.float32)



    def global_pooling(self, p_in_tensor, p_pooling_method = "avg"):
        """Method to perform a global pooling over a set of features.

        Args:
            p_in_tensor (tensor pxd): Tensor to pool.
            p_pooling_method (string): Pooling method (avg, max, min)

        Return:
            tensor bxd: Pooled tensor.
        """
        batch_id_indexs = self.batch_ids_.to(torch.int64)
        if p_pooling_method == "max":
            return scatter_max(p_in_tensor, batch_id_indexs, dim=0)[0]
        elif p_pooling_method == "min":
            return scatter_min(p_in_tensor, batch_id_indexs, dim=0)[0]
        elif p_pooling_method == "avg":
            return scatter_mean(p_in_tensor, batch_id_indexs, dim=0)
        elif p_pooling_method == "sum":
            return scatter_add(p_in_tensor, batch_id_indexs, dim=0)


    def global_upsample(self, p_in_tensor):
        """Method to perform a global upsample over a set of features.

        Args:
            p_in_tensor (tensor bxd): Tensor to upsample.

        Return:
            tensor pxd: Upsampled tensor.
        """
        return torch.index_select(p_in_tensor, 0, self.batch_ids_.to(torch.int64))


    def __repr__(self):
        """Method to create a string representation of 
            object.
        """
        return "### Points:\n{}\n"\
                "### Batch Ids:\n{}\n"\
                "### Batch Size:\n{}\n"\
                "### Pdf:\n{}"\
                .format(self.pts_, self.batch_ids_, 
                self.batch_size_, self.pts_pdf_)


    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Method to apply torch functions to the object.
        """
        if kwargs is None:
            kwargs = {}
        args = [a.pts_ if hasattr(a, 'pts_') else a for a in args]
        ret = func(*args, **kwargs)
        return Pointcloud(ret, self.batch_ids_)

