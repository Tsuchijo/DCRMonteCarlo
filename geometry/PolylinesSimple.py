import torch
from .Polylines import PolyLines
from torch import jit
try:
    import pytest
except ImportError:
    pytest = None

# ============================================================================
# JIT-compiled geometry functions for maximum performance
# ============================================================================

@torch.jit.script
def cross_product_2d_jit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled 2D cross product calculation.
    Handles broadcasting for (N, 2) and (2,) tensors.
    """
    if a.dim() == 1 and b.dim() == 2:
        a = a.unsqueeze(0).expand_as(b)
    elif b.dim() == 1 and a.dim() == 2:
        b = b.unsqueeze(0).expand_as(a)
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

@torch.jit.script
def distance_to_polyline_jit(points: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled distance calculation from a point to a polyline.
    
    Args:
        points: Tensor of shape (N, 2) representing polyline vertices
        point: Tensor of shape (2,) representing the query point
    
    Returns:
        Minimum distance from point to polyline
    """
    u = points[1:] - points[:-1]  # Segment vectors
    v = point - points[:-1]       # Vectors from segment starts to point
    
    # Project point onto each segment
    dot_uv = torch.sum(v * u, dim=-1)
    dot_uu = torch.sum(u * u, dim=-1)
    t = torch.clamp(dot_uv / dot_uu, 0.0, 1.0).unsqueeze(1)
    
    # Find closest points on segments
    closest_points = (1.0 - t) * points[:-1] + t * points[1:]
    distances = torch.norm(closest_points - point, dim=-1)
    
    return torch.min(distances)

@torch.jit.script
def is_silhouette_jit(points: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled silhouette detection for polyline vertices.
    
    Args:
        points: Tensor of shape (N, 2) representing polyline vertices
        point: Tensor of shape (2,) representing the query point
    
    Returns:
        Boolean tensor indicating which interior vertices are silhouette points
    """
    # Get consecutive segments (excluding endpoints to avoid out-of-bounds)
    a = points[:-2]   # First points of segments
    b = points[1:-1]  # Middle points (the ones we're testing)
    c = points[2:]    # End points of segments
    
    # Vectors from consecutive points
    ab = b - a
    bc = c - b
    
    # Vectors from segment points to the test point
    ap = point - a
    bp = point - b
    
    # 2D cross products
    cross_ab_ap = cross_product_2d_jit(ab, ap)
    cross_bc_bp = cross_product_2d_jit(bc, bp)
    
    # Check if point is on opposite sides of the two segments
    return cross_ab_ap * cross_bc_bp < 0

@torch.jit.script
def silhouette_distance_jit(points: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled distance calculation to silhouette points.
    
    Args:
        points: Tensor of shape (N, 2) representing polyline vertices
        point: Tensor of shape (2,) representing the query point
    
    Returns:
        Minimum distance to silhouette points (inf if no silhouette points)
    """
    silhouette_mask = is_silhouette_jit(points, point)
    silhouette_points = points[1:-1][silhouette_mask]
    
    if silhouette_points.shape[0] == 0:
        return torch.tensor(float('inf'), dtype=torch.float32)
    else:
        distances = torch.norm(silhouette_points - point, dim=-1)
        return torch.min(distances)

@torch.jit.script
def ray_intersection_jit(points: torch.Tensor, point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled ray-polyline intersection calculation.
    
    Args:
        points: Tensor of shape (N, 2) representing polyline vertices
        point: Tensor of shape (2,) representing ray origin
        direction: Tensor of shape (2,) representing ray direction
    
    Returns:
        Tensor of intersection times for each segment (inf for no intersection)
    """
    a = points[:-1]  # Segment start points
    b = points[1:]   # Segment end points
    u = b - a        # Segment vectors
    w = point - a    # Vectors from segment starts to ray origin
    
    # Calculate intersection parameters using cross products
    d = cross_product_2d_jit(direction.unsqueeze(0).expand_as(u), u)
    s = cross_product_2d_jit(direction.unsqueeze(0).expand_as(w), w) / d
    t = cross_product_2d_jit(u, w) / d
    
    # Check if intersection is within segment and ray
    valid_intersections = (s >= 0.0) & (s <= 1.0) & (t > 0.0)
    inf_val = torch.tensor(float('inf'), dtype=s.dtype)
    intersection_times = torch.where(valid_intersections, s, inf_val)
    
    return intersection_times

@torch.jit.script
def intersect_polylines_jit(points: torch.Tensor, point: torch.Tensor, direction: torch.Tensor, r: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """
    JIT-compiled polyline intersection with distance limit.
    
    Args:
        points: Tensor of shape (N, 2) representing polyline vertices
        point: Tensor of shape (2,) representing ray origin
        direction: Tensor of shape (2,) representing ray direction
        r: Maximum distance to search for intersections
    
    Returns:
        Tuple of (intersection_point, normal_vector, found_intersection)
    """
    # Normalize direction vector
    direction_norm = torch.norm(direction)
    if direction_norm < 1e-10:
        default_normal = torch.zeros(2)
        default_normal[0] = 1.0
        default_normal[1] = 0.0
        return point, default_normal, False
    
    direction_unit = direction / direction_norm
    
    # Add small offset to avoid numerical issues
    point_offset = point + 1e-6 * direction_unit
    
    # Find intersections
    intersection_times = ray_intersection_jit(points, point_offset, direction_unit)
    
    # Filter valid intersections
    finite_times = intersection_times[torch.isfinite(intersection_times)]
    if finite_times.shape[0] == 0:
        zero_normal = torch.zeros(2)
        return point + r * direction_unit, zero_normal, False
    
    min_time = torch.min(finite_times)
    
    if min_time > r or min_time <= 0.0:
        zero_normal = torch.zeros(2)
        return point + r * direction_unit, zero_normal, False
    
    # Find which segment was intersected
    valid_mask = torch.isfinite(intersection_times) & (intersection_times == min_time)
    idx = torch.where(valid_mask)[0][0]
    
    # Calculate normal vector
    segment_start = points[idx]
    segment_end = points[idx + 1]
    segment_vector = segment_end - segment_start
    segment_length = torch.norm(segment_vector)
    
    if segment_length < 1e-10:
        normal = torch.zeros(2)
        normal[0] = 0.0
        normal[1] = 1.0
    else:
        segment_direction = segment_vector / segment_length
        normal = torch.zeros(2)
        normal[0] = -segment_direction[1]
        normal[1] = segment_direction[0]
    
    intersection_point = point_offset + min_time * direction_unit
    return intersection_point, normal, True

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
        Calculate the distance from a point to the polyline using JIT-compiled implementation.
        
        Args:
            point (torch.Tensor): A tensor of shape (2, ) representing the point.
        
        Returns:
            torch.Tensor: The distance from the point to the polyline.
        """
        return distance_to_polyline_jit(self.points, point)

    @staticmethod
    def funcToPolyline(func: callable, x_min: float, x_max: float, resolution: float) -> 'PolyLines':
        """
        Convert a 1D heightmap function to a PolyLines object.
        Args:
            func (callable): A function that takes an x-coordinate and returns the y-coordinate of the surface.
            x_min (float): the min x value for the polyline
            x_max (float): The maximum x value for the polyline.
            resolution (float): The distance between points in the polyline.
        Returns:
            PolyLines: A PolyLines object representing the polyline.
        """
        x = torch.arange(0, x_max, resolution)
        y = func(x)
        return PolyLinesSimple(torch.stack((x, y), dim=-1))
    
    def isSilhouette(self, point: torch.Tensor) -> torch.Tensor:
        """
        Given a point, returns a tensor of boolean values indicating which points are silhouette points using JIT.
        This checks over only the interior points of the polyline, excluding the first and last points.
        
        Args:
            point (torch.Tensor): A tensor of shape (2,) representing the point.
            
        Returns:
            torch.Tensor: A tensor of boolean values indicating which points are silhouette points.
        """
        return is_silhouette_jit(self.points, point)

    def silhouetteDistance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance from a point to the silhouette of the polyline using JIT.
        
        Args:
            point (torch.Tensor): A tensor of shape (2,) representing the point.
        
        Returns:
            torch.Tensor: The distance from the point to the silhouette of the polyline.
        """
        return silhouette_distance_jit(self.points, point)
    
    def crossProduct2D(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculate the 2D cross product of two vectors using JIT-compiled implementation.
        Supports broadcasting if one of a or b is (2,) and the other is (N, 2).
        
        Args:
            a (torch.Tensor): A tensor of shape (N, 2) or (2,) representing the first vector(s).
            b (torch.Tensor): A tensor of shape (N, 2) or (2,) representing the second vector(s).
        
        Returns:
            torch.Tensor: A tensor of shape (N,) representing the 2D cross product of the vectors.
        """
        return cross_product_2d_jit(a, b)

    def rayIntersection(self, point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        Check if a ray from the point in the specified direction intersects with the polyline using JIT.
        
        Args:
            point (torch.Tensor): A tensor of shape (2,) representing the starting point of the ray.
            direction (torch.Tensor): A tensor of shape (2,) representing the direction and velocity of the ray.
        
        Returns:
            torch.Tensor: A tensor of float values representing the time of intersection with the polyline.
        """
        return ray_intersection_jit(self.points, point, direction)
    
    def intersectPolylines(self, point: torch.Tensor, direction: torch.Tensor, r: float) -> torch.Tensor:
        """
        Find the first intersection of a ray from the point in the specified direction with the polyline using JIT.
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
        return intersect_polylines_jit(self.points, point, direction, r)

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