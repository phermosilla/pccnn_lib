import numpy as np
import torch

from pccnn_lib.op_wrappers import FindNeighbors, ComputeKeys

class Neighborhood(object):
    """Class to represent a neighborhood.
    """

    def __init__(self, p_grid, p_samples, p_max_neighbors = 0):
        """Constructor.

        Args:
            p_grid (Grid): Grid.
            p_samples (Pointcloud): Sample point cloud.
            p_max_neighbors (int): Maximum number of neighbors collected.
        """

        # Store variables.
        self.grid_ = p_grid
        self.samples_ = p_samples

        # Compute the cell ids per sample.
        with torch.no_grad():
            
            # Sort the points of the grid.
            sorted_pts = torch.index_select(
                        self.grid_.pointcloud_.pts_, 0, 
                        self.grid_.sorted_ids_.to(torch.int64))
            sorted_pts_cell_ids = torch.index_select(
                        self.grid_.cell_ids_, 0, 
                        self.grid_.sorted_ids_.to(torch.int64))

            # Compute keys for the samples.
            sample_cell_ids = ComputeKeys.apply(self.samples_.pts_, 
                self.samples_.batch_ids_, self.grid_.bounding_box_.min_,
                self.grid_.num_cells_, self.grid_.cell_size_)

            # Compute the neighbors.
            neighbors, self.start_ids_ = FindNeighbors.apply(
                sorted_pts, sorted_pts_cell_ids, 
                self.samples_.pts_, sample_cell_ids, 
                self.grid_.grid_ds_, self.grid_.num_cells_,
                self.grid_.cell_size_, p_max_neighbors)

            # Compute the neighbor ids to the original point cloud.
            orig_neighs = torch.index_select(
                        self.grid_.sorted_ids_, 0, 
                        neighbors[:,0].to(torch.int64)).to(torch.int32)
            self.neighbors_ = torch.cat((
                torch.reshape(orig_neighs, [-1, 1]), 
                torch.reshape(neighbors[:,1], [-1, 1])),
                -1)


    def __repr__(self):
        """Method to create a string representation of 
            object.
        """
        return "### Neighbors:\n{}\n"\
                "### Start indices:\n{}"\
                .format(self.neighbors_, self.start_ids_)
