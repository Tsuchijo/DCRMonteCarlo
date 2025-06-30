import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

        self.boundaryDirichlet = lambda x: 0.0 # default boundary condition for Dirichlet boundaries returns 0

        self.surface_topo_polyline = PolyLines.funcToPolyline(surface_topo, boundary_size[0], grid_res[0])
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
    
    def set_boundary_conditions(self, boundaryDirichlet: callable):
        """
        Set the boundary conditions for the solver.
        
        Args:
            boundaryDirichlet (callable): A function that takes a point and returns the value of the Dirichlet boundary condition at that point.
        """
        self.boundaryDirichlet = boundaryDirichlet

    def set_solve_points(self, solve_points: torch.Tensor):
        """
        Set the points where the solution is to be evaluated.
        
        Args:
            solve_points (torch.Tensor): A tensor of points where the solution is to be evaluated.
        """
        self.solve_points = solve_points

    def solve(self, nWalks = 1000, maxSteps = 1000) -> torch.Tensor:
        """
        Solve the forward problem for the given surface topology and boundary size.
        
        Returns:
            torch.Tensor: A tensor of shape (N, 2) where N is the number of points in the solve_points tensor.
        """

        eps = 1e-4 # stopping tolerance
        rmin = 1e-3 # Minimum step size for the random walk
        results = torch.zeros((len(self.solve_points), 1)) # Initialize results tensor to store the accumulated solution values
        for i, point in enumerate(tqdm(self.solve_points, desc="Solving WoS", unit="pt")):
            # Initialize the random walk
            current_point = point.clone()
            for _ in range(nWalks):
                step_count = 0
                dDirichlet = 1.0 # Distance to the Dirichlet boundary
                onBoundary = False # Flag to check if the point is on the boundary
                nomral = torch.tensor([1.0, 0.0]) # Normal vector at the intersection point
                while (step_count < maxSteps) & (dDirichlet > eps):
                    dDirichlet = self.outer_boundary_polyline.distance(current_point)
                    dNeumann = self.surface_topo_polyline.silhouetteDistance(current_point)
                    r = max(rmin, min(dDirichlet, dNeumann)) # Step size is the minimum distance to the boundaries
                    theta = torch.rand(1) * 2 * np.pi # Random direction
                    if onBoundary:
                        theta = theta/2 + torch.atan2(nomral[1], nomral[0]) # Reflect the direction if on the boundary
                    direction = torch.tensor([torch.cos(theta), torch.sin(theta)]) # Direction vector
                    current_point, nomral, onBoundary = self.surface_topo_polyline.intersectPolylines(current_point, direction, r) # Get the next point and normal vector
                
                results[i] += self.boundaryDirichlet(current_point) # Accumulate the solution value at the point

        return results / nWalks # Return the average solution value at each point

    
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
            point (torch.Tensor): A tensor of shape (2, ) representing the point.
        
        Returns:
            torch.Tensor: The distance from the point to the polyline.
        """
        
        return torch.min(torch.norm(self.points - point, dim=-1))

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
        # add a slight offset to the point to avoid numerical issues with the ray intersection
        point = point + 1e-6 * direction / torch.norm(direction)

        intersection_times = self.rayIntersection(point, direction)
        min_time = torch.min(intersection_times)
        
        if min_time < r:
            return point + min_time * direction, torch.tensor([0, 0]), False
        else:
            # Get the normal vector of the line segment we are intersecting
            idx = torch.argmin(intersection_times)
            segment_start = self.points[idx]
            segment_end = self.points[idx + 1]
            segment_vector = segment_end - segment_start
            # Normalize the segment vector to get the direction
            normal = segment_vector / torch.norm(segment_vector)
            # rotate the direction by 90 degrees to get the normal vector
            normal = torch.tensor([-normal[1], normal[0]])
            # Return the point on the circle at distance r in the direction of the ray
            return point + r * direction / torch.norm(direction), normal, True

    


def test_wos_solver():
    """Test the Walk on Spheres solver with a simple geometry and alternating boundary conditions."""
    
    # Define a simple wavy surface topology
    def surface_func(x):
        return 1.0 + 0.3 * torch.sin(4 * np.pi * x / 2.0)  # Wavy surface
    
    # Domain size
    boundary_size = (2.0, 2.0)
    grid_res = (0.05, 0.05)  # Higher resolution for better visualization
    
    # Create solver
    solver = WostSolver_2D(surface_func, boundary_size, grid_res=grid_res)
    
    # Define alternating boundary conditions on the outer boundary
    def alternating_boundary(point):
        x, y = point[0], point[1]
        # Create alternating pattern based on position
        if x <= 1e-6:  # Left boundary
            return 1.0 if y < 1.0 else -1.0
        elif x >= boundary_size[0] - 1e-6:  # Right boundary  
            return -1.0 if y < 1.0 else 1.0
        elif y <= 1e-6:  # Bottom boundary
            return 1.0 if x < 1.0 else -1.0
        elif y >= boundary_size[1] - 1e-6:  # Top boundary
            return -1.0 if x < 1.0 else 1.0
        else:
            return 0.0  # Interior (shouldn't be called)
    
    solver.set_boundary_conditions(alternating_boundary)
    
    # Solve the problem
    print("Starting Walk on Spheres solution...")
    solution = solver.solve(nWalks=50, maxSteps=500)  # Reduced for faster computation
    
    # Reshape solution for plotting
    nx = int(boundary_size[0] / grid_res[0])
    ny = int(boundary_size[1] / grid_res[1])
    solution_grid = solution.reshape(nx, ny)
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Geometry and boundary conditions
    x_coords = torch.arange(0, boundary_size[0] + grid_res[0], grid_res[0])
    y_surface = surface_func(x_coords)
    
    ax1.plot(x_coords, y_surface, 'k-', linewidth=2, label='Internal Surface')
    
    # Draw outer boundary with colors indicating boundary conditions
    boundary_points = [
        ([0, 0], [boundary_size[0], 0], 'bottom'),
        ([boundary_size[0], 0], [boundary_size[0], boundary_size[1]], 'right'),
        ([boundary_size[0], boundary_size[1]], [0, boundary_size[1]], 'top'),
        ([0, boundary_size[1]], [0, 0], 'left')
    ]
    
    colors = {'bottom': ['red', 'blue'], 'right': ['blue', 'red'], 
              'top': ['blue', 'red'], 'left': ['red', 'blue']}
    
    for (start, end, side) in boundary_points:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # First half
        if side in ['bottom', 'top']:
            ax1.plot([start[0], mid_x], [start[1], mid_y], colors[side][0], linewidth=4)
            ax1.plot([mid_x, end[0]], [mid_y, end[1]], colors[side][1], linewidth=4)
        else:
            ax1.plot([start[0], mid_x], [start[1], mid_y], colors[side][0], linewidth=4)
            ax1.plot([mid_x, end[0]], [mid_y, end[1]], colors[side][1], linewidth=4)
    
    ax1.set_xlim(-0.1, boundary_size[0] + 0.1)
    ax1.set_ylim(-0.1, boundary_size[1] + 0.1)
    ax1.set_aspect('equal')
    ax1.set_title('Geometry and Boundary Conditions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add text annotations for boundary values
    ax1.text(-0.05, 0.5, '+1', fontsize=12, color='red', ha='right', va='center')
    ax1.text(-0.05, 1.5, '-1', fontsize=12, color='blue', ha='right', va='center')
    ax1.text(2.05, 0.5, '-1', fontsize=12, color='blue', ha='left', va='center')
    ax1.text(2.05, 1.5, '+1', fontsize=12, color='red', ha='left', va='center')
    ax1.text(0.5, -0.05, '+1', fontsize=12, color='red', ha='center', va='top')
    ax1.text(1.5, -0.05, '-1', fontsize=12, color='blue', ha='center', va='top')
    ax1.text(0.5, 2.05, '-1', fontsize=12, color='blue', ha='center', va='bottom')
    ax1.text(1.5, 2.05, '+1', fontsize=12, color='red', ha='center', va='bottom')
    
    # Plot 2: Solution heatmap
    x_grid = torch.arange(grid_res[0]/2, boundary_size[0], grid_res[0])
    y_grid = torch.arange(grid_res[1]/2, boundary_size[1], grid_res[1])
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    
    im = ax2.contourf(X, Y, solution_grid, levels=20, cmap='RdBu_r')
    ax2.plot(x_coords, y_surface, 'k-', linewidth=2)
    ax2.set_xlim(0, boundary_size[0])
    ax2.set_ylim(0, boundary_size[1])
    ax2.set_aspect('equal')
    ax2.set_title('WoS Solution')
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: 3D surface plot
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X.numpy(), Y.numpy(), solution_grid.numpy(), 
                           cmap='RdBu_r', alpha=0.8)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Solution')
    ax3.set_title('3D Solution Surface')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Solution statistics:")
    print(f"  Min value: {solution.min():.4f}")
    print(f"  Max value: {solution.max():.4f}")
    print(f"  Mean value: {solution.mean():.4f}")
    print(f"  Std deviation: {solution.std():.4f}")
    
    return solver, solution, solution_grid

if __name__ == "__main__":
    solver, solution, solution_grid = test_wos_solver()


        