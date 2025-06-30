import torch
import numpy as np

class WostSolver_2D:
    """
    A class to solve the forward problem for a 2D surface topology using the Walk on Spheres method.
    This class is designed to work with a 2D surface topology defined by a function and a boundary size.
    This generally solves for mixed boundary conditions for any elliptic pde defined by a laplace equation with a source, diffusion, and absorption term.
    """

    def __init__(self, surface_topo: callable, boundary_size: tuple[float, float], solve_points: torch.Tensor = None, grid_res: tuple[float, float] = (0.1, 0.1)):
        """
        Initialize the solver with the surface topology and boundary size.
        
        Args:
            surface_topo (callable): A function that takes an x-coordinate and returns the y-coordinate of the surface topology.
            boundary_size (tuple[float, float]): The size of the boundary in the x and y dimensions.
            solve_points (torch.Tensor, optional): A tensor of points where the solution is to be evaluated. If None, a grid will be created based on the boundary size and resolution.
            grid_res (tuple[float, float]): The resolution of the grid in the x and y dimensions, this is used for visualization.
        """
        self.surface_topo = surface_topo
        self.boundary_size = boundary_size
        if solve_points is None:
            self.solve_points = self.create_grid_points_2d(boundary_size, grid_res)
        else:
            self.solve_points = solve_points
        self.grid_res = grid_res

        self.surface_topo_polyline = PolyLines.func_to_polyline(surface_topo, boundary_size[0], grid_res[0])
        self.outer_boundary_polyline = PolyLines(torch.tensor([[0, 0], [boundary_size[0], 0], [boundary_size[0], boundary_size[1]], [0, boundary_size[1]], [0, 0]]))

    def create_grid_points_2d(self, grid_size: tuple[float, float], grid_res: tuple[float, float]) -> torch.Tensor:
        """
        Create a uniform grid of points in the specified range, evenly spaced according to the resolution.
        Args:
            grid_size (tuple[float, float]): The size of the grid in the x and y dimensions.
            grid_res (tuple[float, float]): The resolution of the grid in the x and y dimensions.
        """

        x = torch.arange(0, grid_size[0], grid_res[0])
        y = torch.arange(0, grid_size[1], grid_res[1])
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        return torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    
    def distance_to_surface(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance from each point to the surface defined by the surface topology.
        
        Args:
            points (torch.Tensor): A tensor of points where the distance to the surface is to be calculated.
        
        Returns:
            torch.Tensor: A tensor of distances from each point to the surface.
        """
        x = points[:, 0]
        y_surface = self.surface_topo(x)
        return points[:, 1] - y_surface
    
class PolyLines:
    """
    A class to represent a collection of polylines in 2D space.
    Each polyline is defined by a sequence of points.
    """

    def __init__(self, points: torch.Tensor):
        """
        Initialize the PolyLines with a tensor of points.
        
        Args:
            points (torch.Tensor): A tensor of shape (N, 2) where N is the number of points.
        """
        self.points = points

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx]
    
    def distance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance from a point to the polyline.
        
        Args:
            point (torch.Tensor): A tensor of shape (..., 2) representing the point(s).
        
        Returns:
            torch.Tensor: The distance from the point(s) to the polyline.
        """
        original_shape = point.shape[:-1]
        
        # Flatten batch dimensions for easier processing
        if point.dim() > 1:
            point_flat = point.view(-1, 2)
        else:
            point_flat = point.unsqueeze(0)
        
        # Calculate distances from each query point to all polyline points
        # point_flat: (B, 2), self.points: (N, 2) -> distances: (B, N)
        distances = torch.norm(
            point_flat.unsqueeze(1) - self.points.unsqueeze(0), 
            dim=2
        )
        
        # Find minimum distance for each query point
        min_distances = torch.min(distances, dim=1)[0]  # (B,)
        
        # Reshape back to original batch shape
        return min_distances.view(original_shape)
    
    def func_to_polyline(func: callable, x_max: float, resolution: float) -> 'PolyLines':
        """
        Convert a 1D heightmap function to a PolyLines object.
        Args:
            func (callable): A function that takes an x-coordinate and returns the y-coordinate of the surface.
            x_max (float): The maximum x value for the polyline.
            resolution (float): The distance between points in the polyline.
        Returns:
            PolyLines: A PolyLines object representing the polyline.
        """
        x = torch.arange(0, x_max, resolution)
        y = func(x)
        return PolyLines(torch.stack((x, y), dim=-1))
    
    def silhouetteDistance(self, point: torch.Tensor) -> torch.Tensor:
        """
        For a batch of points, for each find the distance to the closest silhouette point.
        
        Args:
            point (torch.Tensor): A tensor of shape (2,) representing the point.
        
        Returns:
            torch.Tensor: The distance from the point to the silhouette of the polyline. 
                   If the point is outside the silhouette, returns a positive distance.
                   If the point is inside the silhouette, returns a negative distance.
        """


        