import numpy as np
import torch

from pccnn_lib.op_wrappers import BuildGridDS, ComputeKeys

class Grid(object):
    """Class to distribute points into a regular grid.
    """

    def __init__(self, p_point_cloud, p_bounding_box, p_cell_size):
        """Constructor.

        Args:
            p_point_cloud (Pointcloud): Input point cloud.
            p_bounding_box (Bounding box): Input bounding box.
            p_cell_size (float): Cell size.
        """
        
        # Save the parameters.
        self.pointcloud_ = p_point_cloud
        self.bounding_box_ = p_bounding_box
        self.cell_size_ = torch.as_tensor(p_cell_size, 
            device=self.pointcloud_.pts_.device)

        # Compute the number of cells.
        with torch.no_grad():
            num_cells = (self.bounding_box_.max_ - self.bounding_box_.min_)/self.cell_size_
            self.num_cells_ = torch.max(num_cells.to(torch.int32) + 1, dim=0)[0]

        # Initialize the grid tensors.
        self.cell_ids_ = None
        self.sorted_ids_ = None
        self.grid_ds_ = None


    def compute_cell_ids(self):
        """ Method to compute the cell ids.
        """

        if self.cell_ids_ is None:

            # Compute the cell id per object.
            with torch.no_grad():

                # Compute cell ids.
                self.cell_ids_ = ComputeKeys.apply(self.pointcloud_.pts_, 
                    self.pointcloud_.batch_ids_, self.bounding_box_.min_,
                    self.num_cells_, self.cell_size_)


                # Sort the indices.
                self.sorted_ids_ = torch.argsort(self.cell_ids_)

    
    def build_ds(self):
        """Method to build the data structure for fast access.
        """

        # Check for errors.
        if self.cell_ids_ is None or self.sorted_ids_ is None:
            raise RuntimeError('In order to build the data structure first the'
                ' keys for each point has to be computed.')

        # Create the data structure.
        if self.grid_ds_ is None:
            
            with torch.no_grad():
                sorted_keys = torch.index_select(
                        self.cell_ids_, 0, 
                        self.sorted_ids_.to(torch.int64))
                self.grid_ds_ = BuildGridDS.apply(sorted_keys, 
                    self.num_cells_, self.pointcloud_.batch_size_)


    def __repr__(self):
        """Method to create a string representation of 
            object.
        """
        return "### Cell size:\n{}\n"\
                "### Num cells:\n{}\n"\
                "### Cell Ids:\n{}\n"\
                "### Sorted Ids:\n{}\n"\
                "### Grid Ds:\n{}"\
                .format(self.cell_size_, 
                self.num_cells_, self.cell_ids_,
                self.sorted_ids_, self.grid_ds_)