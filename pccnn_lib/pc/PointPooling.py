import numpy as np
import torch
from torch_scatter import scatter_max, scatter_mean

from pccnn_lib.pc.Pointcloud import Pointcloud
from pccnn_lib.pc.BoundingBox import BoundingBox
from pccnn_lib.pc.Grid import Grid
from pccnn_lib.pc.Neighborhood import Neighborhood
from pccnn_lib.op_wrappers import FindNeighbors
from pccnn_lib.op_wrappers import PoissonDiskSamlping

class PointPooling(object):
    """Class to represent a pooling operation for a point cloud.
    """

    def __init__(self, 
        p_point_cloud, 
        p_radii, 
        p_method = "poisson_disk"):
        """Constructor.

        Args:
            p_point_cloud (Pointcloud): Input point cloud.
            p_radii (tensor d): Poolint radii
            p_method (string): Pooling method:
                - poisson_disk: Poisson disk.
                - cell_average: Average of the cell.
                - indices: Pool using given indices.
        """
        
        # Save parameters.
        self.point_cloud_ = p_point_cloud
        self.radii_ = p_radii
        self.method_ = p_method

        # Compute the pooling.
        with torch.no_grad():

            if p_method == "poisson_disk":
                
                # Compute bounding box.
                bb = BoundingBox(self.point_cloud_)

                # Compute grid.
                grid = Grid(self.point_cloud_, bb, self.radii_)
                grid.compute_cell_ids()
                grid.build_ds()

                # Compute neighborhood.
                sorted_pts = torch.index_select(
                            self.point_cloud_.pts_, 0, 
                            grid.sorted_ids_.to(torch.int64))
                sorted_pts_cell_ids = torch.index_select(
                            grid.cell_ids_, 0, 
                            grid.sorted_ids_.to(torch.int64))

                # Compute the neighbors.
                neighbors, start_ids = FindNeighbors.apply(
                    sorted_pts, sorted_pts_cell_ids, 
                    sorted_pts, sorted_pts_cell_ids, 
                    grid.grid_ds_, grid.num_cells_,
                    grid.cell_size_, 0)

                # Compute the sampling.
                sampled_pts_ids = PoissonDiskSamlping.apply(sorted_pts_cell_ids, 
                    neighbors, start_ids, grid.num_cells_)
                sampled_pts_ids = torch.reshape(sampled_pts_ids, [-1])
                self.pooling_ids_ = torch.index_select(
                    grid.sorted_ids_, 0, sampled_pts_ids.to(torch.int64)).to(torch.int64)
                
            elif p_method == "cell_average":

                # Compute bounding box.
                bb = BoundingBox(self.point_cloud_)
                
                # Compute grid.
                grid = Grid(self.point_cloud_, bb, self.radii_)
                grid.compute_cell_ids()
                grid.build_ds()

                unique_ids, inverse_ids = torch.unique(grid.cell_ids_, return_inverse=True)
                self.pooling_ids_ = inverse_ids
                self.num_pooling_pts_ = unique_ids.shape[0]

            
    def pool_tensor(self, p_in_tensor, max_pool = False):
        """Method to pool a tensor using indices computed in the constructor.

        Args:
            p_in_tensor (Tensor nxl): Input tensor.
        Returns:
            Tensor mxl: Pooled tensor.
        """
        if self.method_ == "poisson_disk":
            return torch.index_select(
                p_in_tensor, 0, self.pooling_ids_)
        elif self.method_ == "cell_average":
            pooling_ids_int64 = self.pooling_ids_.to(torch.int64)
            if max_pool:
                return scatter_max(p_in_tensor, pooling_ids_int64, dim=0)[0]
            else:
                return scatter_mean(p_in_tensor, pooling_ids_int64, dim=0)
            