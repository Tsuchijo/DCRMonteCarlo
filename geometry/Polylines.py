import torch
from torch import jit
try:
    import pytest
except ImportError:
    pytest = None

class PolyLines:
    """
    A template class to represent a collection of polylines in 2D space.
    Each polyline is defined by a sequence of points. For walk on stars we requires functions to calculate silhouettes and ray intersections and distances.
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
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def isSilhouette(self, point: torch.Tensor) -> torch.Tensor:
        """
        Given a point, returns a tensor of boolean values indicating which points are silhouette points.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def silhouetteDistance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance from a point to the silhouette of the polyline.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def rayIntersection(self, point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Check if a ray from the point in the specified direction intersects with the polyline.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def intersectPolylines(self, point: torch.Tensor, direction: torch.Tensor, r: float) -> torch.Tensor:
        """
        Find the first intersection of a ray from the point in the specified direction with the polyline which is within a distance r.
        If no intersection is found within the distance r, the point on the circle at distance r in the direction of the ray is returned.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")