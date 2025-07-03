import torch
import pytest
from torch import jit

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
    

class PolyLinesSimple(PolyLines):
    """
    A class to represent a collection of polylines in 2D space, uses a naive approach to calculate silhouettes and ray intersections.
    Each polyline is defined by a sequence of points.
    """

    def __init__(self, points: torch.Tensor):
        """
        Initialize the PolyLines with a tensor of points.
        
        Args:
            points (torch.Tensor): A tensor of shape (N, 2) where N is the number of points.
        """
        super().__init__(points)
    
    def distance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance from a point to the polyline.
        
        Args:
            point (torch.Tensor): A tensor of shape (2, ) representing the point.
        
        Returns:
            torch.Tensor: The distance from the point to the polyline.
        """
        
        u =  self.points[1:] - self.points[:-1]  # Vector along the polyline segments
        # Compute the projection factor t for each segment in a batched way
        v = point - self.points[:-1]  # (num_segments, 2)
        t = torch.sum(v * u, dim=-1) / torch.sum(u * u, dim=-1)  # Batched dot product and squared norm
        t = torch.clamp(t, 0, 1).unsqueeze(1)  # Clamp t to the segment range [0, 1]
        
        closest_points = (1.0 - t) * self.points[:-1] + t * self.points[1:]  # Interpolated points on the segments
        distances = torch.norm(closest_points - point, dim=-1)  # Distances from the point to the closest points on the segments
        return torch.min(distances)  # Return the minimum distance

    @staticmethod
    def funcToPolyline(func: callable, x_max: float, resolution: float) -> 'PolyLines':
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
    
    def isSilhouette(self, point: torch.Tensor) -> torch.Tensor:
        """
        Given a point, returns a tensor of boolean values indicating which points are silhouette points.
        This checks over only the interior points of the polyline, excluding the first and last points.
        
        Args:
            point (torch.Tensor): A tensor of shape (2,) representing the point.
            
        Returns:
            torch.Tensor: A tensor of boolean values indicating which points are silhouette points.
        """
        # Get consecutive segments (excluding the last segment to avoid out-of-bounds)
        a = self.points[:-2]  # First points of segments
        b = self.points[1:-1] # Middle points (the ones we're testing)
        c = self.points[2:]   # End points of segments
        
        # Vectors from a to b and b to c
        ab = b - a
        bc = c - b
        
        # Vectors from segment points to the test point
        ap = point - a
        bp = point - b
        
        # 2D cross products (returns scalars)
        cross_ab_ap = self.crossProduct2D(ab, ap)
        cross_bc_bp = self.crossProduct2D(bc, bp)
        
        # Check if point is on opposite sides of the two segments
        return cross_ab_ap * cross_bc_bp < 0

    def silhouetteDistance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance from a point to the silhouette of the polyline.
        
        Args:
            point (torch.Tensor): A tensor of shape (2,) representing the point.
        
        Returns:
            torch.Tensor: The distance from the point to the silhouette of the polyline.
        """
        # Get silhouette points
        silhouette_points = self.points[1:-1][self.isSilhouette(point)]
        
        if silhouette_points.shape[0] == 0:
            return torch.tensor(float('inf'))
        else:
            # Calculate distances to silhouette points
            distances = torch.norm(silhouette_points - point, dim=-1)
            return torch.min(distances)
    
    def crossProduct2D(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculate the 2D cross product of two vectors.
        Supports broadcasting if one of a or b is (2,) and the other is (N, 2).
        
        Args:
            a (torch.Tensor): A tensor of shape (N, 2) or (2,) representing the first vector(s).
            b (torch.Tensor): A tensor of shape (N, 2) or (2,) representing the second vector(s).
        
        Returns:
            torch.Tensor: A tensor of shape (N,) representing the 2D cross product of the vectors.
        """
        if a.dim() == 1 and b.dim() == 2:
            a = a.unsqueeze(0).expand_as(b)
        elif b.dim() == 1 and a.dim() == 2:
            b = b.unsqueeze(0).expand_as(a)
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    def rayIntersection(self, point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Check if a ray from the point in the specified direction intersects with the polyline.
        
        Args:
            point (torch.Tensor): A tensor of shape (2,) representing the starting point of the ray.
            direction (torch.Tensor): A tensor of shape (2,) representing the direction and velocity of the ray.
        
        Returns:
            torch.Tensor: A tensor of float values representing the time of intersection with the polyline.
        """
        
        a = self.points[:-1]
        b = self.points[1:]
        u = b - a
        w = point - a
        d = self.crossProduct2D(direction, u)
        s = self.crossProduct2D(direction, w) / d
        t = self.crossProduct2D(u, w) / d
        
        # Check if the intersection is within the segment and the ray
        valid_intersections = (s >= 0) & (s <= 1) & (t > 0)
        intersection_times = torch.where(valid_intersections, s, torch.tensor(float('inf')))
        return intersection_times
    
    def intersectPolylines(self, point: torch.Tensor, direction: torch.Tensor, r: float) -> torch.Tensor:
        """
        Find the first intersection of a ray from the point in the specified direction with the polyline which is within a distance r.
        If no intersection is found within the distance r, the point on the circle at distance r in the direction of the ray is returned.
        Args:
            point (torch.Tensor): A tensor of shape (2,) representing the starting point of the ray.
            direction (torch.Tensor): A tensor of shape (2,) representing the direction and velocity of the ray.
            r (float): The distance within which to find the intersection.
        Returns:
            torch.Tensor: A tensor of shape (2,) representing the point of intersection or the point on the circle at distance r in the direction of the ray.
            torch.Tensor: A tensor of shape (2,) representing the normal vector at the intersection point, or a zero vector if no intersection is found.
            bool: A boolean indicating whether an intersection was found.
        """
        # Normalize direction vector to ensure unit length
        direction_norm = torch.norm(direction)
        if direction_norm < 1e-10:
            # Handle zero direction vector
            return point, torch.tensor([1.0, 0.0]), False
            
        direction_unit = direction / direction_norm
        
        # Add a slight offset to avoid numerical issues with the ray intersection
        point_offset = point + 1e-6 * direction_unit

        intersection_times = self.rayIntersection(point_offset, direction_unit)
        
        # Check if we have any valid intersections
        finite_times = intersection_times[torch.isfinite(intersection_times)]
        if len(finite_times) == 0:
            # No intersection found, return point at distance r
            return point + r * direction_unit, torch.tensor([0.0, 0.0]), False
        
        min_time = torch.min(finite_times)
        
        if min_time > r or min_time <= 0:
            # Intersection is beyond our search radius or behind us
            return point + r * direction_unit, torch.tensor([0.0, 0.0]), False
        else:
            # Find which segment we intersected
            valid_intersections = torch.isfinite(intersection_times) & (intersection_times == min_time)
            idx = torch.where(valid_intersections)[0][0].item()  # Get first valid intersection
            
            # Get the normal vector of the line segment we are intersecting
            segment_start = self.points[idx]
            segment_end = self.points[idx + 1]
            segment_vector = segment_end - segment_start
            
            # Handle zero-length segments
            segment_length = torch.norm(segment_vector)
            if segment_length < 1e-10:
                normal = torch.tensor([0.0, 1.0])  # Default normal
            else:
                # Normalize the segment vector to get the direction
                segment_direction = segment_vector / segment_length
                # Rotate by 90 degrees to get the normal vector (pointing outward)
                normal = torch.tensor([-segment_direction[1], segment_direction[0]])
            
            # Return the intersection point and normal
            intersection_point = point_offset + min_time * direction_unit
            return intersection_point, normal, True

def test_polyline_distance():
    # Create a square domain with polylines
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    polyline = PolyLinesSimple(points)
    point = torch.tensor([0.5, 0.5])
    distance = polyline.distance(point)
    expected_distance = 0.5 
    assert torch.isclose(distance, torch.tensor(expected_distance), atol=1e-6), f"Expected {expected_distance}, got {distance.item()}"
    
def test_polyline_silhouette():
    # Create a simple polyline
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    polyline = PolyLinesSimple(points)
    point = torch.tensor([1.5, 0.6])
    silhouette = polyline.isSilhouette(point)
    expected_silhouette = torch.tensor([ True ])
    assert torch.equal(silhouette, expected_silhouette), f"Expected {expected_silhouette}, got {silhouette}"

def test_polyline_silhouette_distance():
    # Create a simple polyline
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    polyline = PolyLinesSimple(points)
    point = torch.tensor([1.5, 0.6])
    distance = polyline.silhouetteDistance(point)
    expected_distance = torch.norm(torch.tensor([1.5, 0.6]) - torch.tensor([1.0, 1.0]))
    assert torch.isclose(distance, expected_distance, atol=1e-6), f"Expected {expected_distance}, got {distance.item()}"

def test_polyline_ray_intersection():
    # Create a simple polyline
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    polyline = PolyLinesSimple(points)
    point = torch.tensor([0.5, 0.5])
    direction = torch.tensor([1.0, 0.0])
    intersection_times = polyline.rayIntersection(point, direction)
    expected_times = torch.tensor([float('inf'), 0.5, float('inf'), float('inf')])
    assert torch.allclose(intersection_times, expected_times, atol=1e-6), f"Expected {expected_times}, got {intersection_times}"

def test_polyline_intersect_polylines():
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]) # square domain
    polyline = PolyLinesSimple(points)
    point = torch.tensor([0.5, 0.5])
    direction = torch.tensor([1.0, 0.0])  # ray pointing
    r = 2.0  # distance within which to find the intersection
    intersection_point, normal, found = polyline.intersectPolylines(point, direction, r)
    expected_intersection = torch.tensor([1.0, 0.5])
    expected_normal = torch.tensor([-1.0, 0.0])  # normal
    assert torch.allclose(intersection_point, expected_intersection, atol=1e-6), f"Expected intersection point {expected_intersection}, got {intersection_point}"
    assert torch.allclose(normal, expected_normal, atol=1e-6), f"Expected normal {expected_normal}, got {normal}"
    assert found == True, "Expected intersection to be found"

if __name__ == "__main__":
    # Test each method in the PolyLinesSimple class using pytest
    pytest.main([__file__])
else:
    # If this file is imported, do not run the tests
    pass