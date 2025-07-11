import torch
import pytest
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Dict, List, Optional, Tuple, Any

def torchGradient(function: callable, point: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of a function at a given point using PyTorch's autograd.
    
    Args:
        function (callable): A function that takes a point and returns a scalar value.
        point (torch.Tensor): The point at which to compute the gradient.
    
    Returns:
        torch.Tensor: The gradient of the function at the given point.
    """
    # Ensure point requires gradients
    if not point.requires_grad:
        point = point.clone().requires_grad_(True)
    
    value = function(point)
    
    # Ensure the function output is a scalar
    if value.numel() != 1:
        raise ValueError(f"Function must return a scalar, got tensor with {value.numel()} elements")
    
    gradient = torch.autograd.grad(value, point, create_graph=True)[0]
    return gradient

def torchLaplacian(function: callable, point: torch.Tensor) -> torch.Tensor:
    """
    Compute the Laplacian of a function at a given point using PyTorch's autograd.
    
    Args:
        function (callable): A function that takes a point and returns a scalar value.
        point (torch.Tensor): The point at which to compute the Laplacian.
    
    Returns:
        torch.Tensor: The Laplacian of the function at the given point.
    """
    # Ensure point requires gradients
    if not point.requires_grad:
        point = point.clone().requires_grad_(True)
    
    # Compute first derivatives (gradient)
    gradient = torchGradient(function, point)
    
    # Compute second derivatives (Laplacian) by taking divergence of gradient
    laplacian = torch.zeros_like(gradient[0]) + 1e-8 # add small value to prevent divergence
    try:
        for i in range(len(gradient)):
            # Compute second derivative with respect to each dimension
            second_deriv = torch.autograd.grad(gradient[i], point, create_graph=True, retain_graph=True)[0][i]
            laplacian = laplacian + second_deriv
    except Exception as e:
        return laplacian
        
    return laplacian

def gridSampleMinMax(function: callable, domain_bounds: list, grid_resolution: int = 100) -> tuple:
    """
    Find the minimum and maximum values of a function over a rectangular domain using grid sampling.
    
    Args:
        function (callable): A function that takes a torch.Tensor point and returns a scalar value.
        domain_bounds (list): List of [min, max] bounds for each dimension. 
                             E.g., [[x_min, x_max], [y_min, y_max]] for 2D.
        grid_resolution (int): Number of grid points per dimension.
    
    Returns:
        tuple: (min_value, max_value, min_point, max_point) where min/max_point are torch.Tensors
    """
    ndim = len(domain_bounds)
    
    # Create grid points for each dimension
    grid_1d = []
    for bounds in domain_bounds:
        grid_1d.append(torch.linspace(bounds[0], bounds[1], grid_resolution))
    
    # Create meshgrid for all dimensions
    if ndim == 1:
        grid_points = grid_1d[0].unsqueeze(1)
    elif ndim == 2:
        x_grid, y_grid = torch.meshgrid(grid_1d[0], grid_1d[1], indexing='ij')
        grid_points = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
    elif ndim == 3:
        x_grid, y_grid, z_grid = torch.meshgrid(grid_1d[0], grid_1d[1], grid_1d[2], indexing='ij')
        grid_points = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)
    else:
        raise ValueError(f"Grid sampling for {ndim}D not implemented. Maximum supported dimension is 3.")
    
    # Evaluate function at all grid points
    values = []
    for point in grid_points:
        try:
            val = function(point)
            if torch.isnan(val) or torch.isinf(val):
                continue  # Skip invalid values
            values.append(val.item() if hasattr(val, 'item') else float(val))
        except:
            continue  # Skip points where function evaluation fails
    
    if not values:
        raise ValueError("Function could not be evaluated at any grid points")
    
    values = torch.tensor(values)
    min_idx = torch.argmin(values)
    max_idx = torch.argmax(values)
    
    min_value = values[min_idx]
    max_value = values[max_idx]
    min_point = grid_points[min_idx]
    max_point = grid_points[max_idx]
    
    return min_value.item(), max_value.item(), min_point, max_point


def torch_smooth_circle(x: torch.Tensor, center, radius):
    """
    Define a smooth circle function which can be differetiated for building spatially varying field
    returns 0 outside the circle and 1 inside the circle with smooth differentiable transition using sigmoid 
    """
    sdf = ((x - center).norm() - radius) # negativen inside the circle positive outside
    return (-100 * sdf).sigmoid()



def test_torchGradient():
    """Test torchGradient function with known analytical derivatives."""
    
    # Test 1: Simple quadratic f(x) = x^2, gradient should be 2x
    def quadratic_1d(x):
        return x ** 2
    
    point_1d = torch.tensor([3.0], requires_grad=True)
    grad = torchGradient(quadratic_1d, point_1d)
    expected = torch.tensor([6.0])  # 2 * 3
    assert torch.allclose(grad, expected, atol=1e-6), f"Expected {expected}, got {grad}"
    
    # Test 2: Multivariable f(x,y) = x^2 + y^2, gradient should be [2x, 2y]
    def quadratic_2d(point):
        x, y = point[0], point[1]
        return x ** 2 + y ** 2
    
    point_2d = torch.tensor([2.0, 3.0], requires_grad=True)
    grad = torchGradient(quadratic_2d, point_2d)
    expected = torch.tensor([4.0, 6.0])  # [2*2, 2*3]
    assert torch.allclose(grad, expected, atol=1e-6), f"Expected {expected}, got {grad}"
    
    # Test 3: Linear function f(x) = 3x + 2, gradient should be 3
    def linear(x):
        return 3 * x + 2
    
    point_linear = torch.tensor([5.0], requires_grad=True)
    grad = torchGradient(linear, point_linear)
    expected = torch.tensor([3.0])
    assert torch.allclose(grad, expected, atol=1e-6), f"Expected {expected}, got {grad}"
    
    print("torchGradient tests passed!")

def test_torchLaplacian():
    """Test torchLaplacian function with known analytical Laplacians."""
    
    # Test 1: f(x,y) = x^2 + y^2, Laplacian should be 4 (2 + 2)
    def quadratic_2d(point):
        x, y = point[0], point[1]
        return x ** 2 + y ** 2
    
    point_2d = torch.tensor([1.0, 2.0], requires_grad=True)
    laplacian = torchLaplacian(quadratic_2d, point_2d)
    expected = torch.tensor(4.0)  # d²/dx² + d²/dy² = 2 + 2
    assert torch.allclose(laplacian, expected, atol=1e-6), f"Expected {expected}, got {laplacian}"
    
    # Test 2: f(x,y,z) = x^2 + y^2 + z^2, Laplacian should be 6 (2 + 2 + 2)
    def quadratic_3d(point):
        x, y, z = point[0], point[1], point[2]
        return x ** 2 + y ** 2 + z ** 2
    
    point_3d = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    laplacian = torchLaplacian(quadratic_3d, point_3d)
    expected = torch.tensor(6.0)  # 2 + 2 + 2
    assert torch.allclose(laplacian, expected, atol=1e-6), f"Expected {expected}, got {laplacian}"
    
    # Test 3: f(x,y) = x^4 + y^4, Laplacian should be 12x^2 + 12y^2
    def quartic_2d(point):
        x, y = point[0], point[1]
        return x ** 4 + y ** 4
    
    point_quartic = torch.tensor([2.0, 1.0], requires_grad=True)
    laplacian = torchLaplacian(quartic_2d, point_quartic)
    expected = torch.tensor(12 * 4 + 12 * 1, dtype=torch.float32)  # 12*2^2 + 12*1^2 = 48 + 12 = 60
    assert torch.allclose(laplacian, expected, atol=1e-6), f"Expected {expected}, got {laplacian}"
    
    print("torchLaplacian tests passed!")

def test_gridSampleMinMax():
    """Test gridSampleMinMax function with known functions."""
    
    # Test 1: Simple 2D quadratic f(x,y) = x^2 + y^2, min at origin
    def quadratic_2d(point):
        x, y = point[0], point[1]
        return x ** 2 + y ** 2
    
    domain_bounds = [[-2.0, 2.0], [-2.0, 2.0]]
    min_val, max_val, min_point, max_point = gridSampleMinMax(quadratic_2d, domain_bounds, 50)
    
    assert abs(min_val) < 0.1, f"Expected min near 0, got {min_val}"
    assert abs(min_point[0]) < 0.1 and abs(min_point[1]) < 0.1, f"Expected min point near origin, got {min_point}"
    
    # Test 2: 1D function f(x) = -x^2 + 4, max at x=0
    def parabola_1d(point):
        x = point[0]
        return -x ** 2 + 4
    
    domain_bounds = [[-3.0, 3.0]]
    min_val, max_val, min_point, max_point = gridSampleMinMax(parabola_1d, domain_bounds, 100)
    
    assert abs(max_val - 4.0) < 0.1, f"Expected max near 4, got {max_val}"
    assert abs(max_point[0]) < 0.1, f"Expected max point near 0, got {max_point[0]}"
    
    print("gridSampleMinMax tests passed!")

def run_all_tests():
    """Run all utility tests."""
    test_torchGradient()
    test_torchLaplacian()
    test_gridSampleMinMax()
    print("All utility tests passed!")



def plot_walk_history(walk_history: Dict[int, List[Dict]], 
                     point_idx: int = 0, 
                     walk_idx: int = 0,
                     polylines: Optional[Dict[str, Any]] = None,
                     figsize: Tuple[int, int] = (12, 8),
                     show_step_circles: bool = True,
                     show_path_line: bool = True,
                     show_contributions: bool = True,
                     step_circle_alpha: float = 0.3,
                     path_line_alpha: float = 0.8,
                     title: Optional[str] = None) -> plt.Figure:
    """
    Plot the full history of a single Monte Carlo walk, showing:
    - All points visited during the walk
    - Circles representing possible step positions (dDirichlet and dNeumann ranges)
    - Optional boundary visualization from polylines objects
    - Contribution points and types
    
    Parameters:
    -----------
    walk_history : Dict[int, List[Dict]]
        Walk history dictionary from solver.solve(..., return_history=True)
        Structure: {point_idx: [walk_data_dict, ...]}
    point_idx : int, default=0
        Index of the solve point to visualize
    walk_idx : int, default=0
        Index of the specific walk to visualize for that point
    polylines : Optional[Dict[str, Any]], default=None
        Dictionary containing boundary polylines objects:
        {'dirichlet': polylines_obj, 'neumann': polylines_obj}
    figsize : Tuple[int, int], default=(12, 8)
        Figure size in inches
    show_step_circles : bool, default=True
        Whether to show circles representing possible step positions
    show_path_line : bool, default=True
        Whether to connect walk points with lines
    show_contributions : bool, default=True
        Whether to highlight contribution points
    step_circle_alpha : float, default=0.3
        Transparency of step circles
    path_line_alpha : float, default=0.8
        Transparency of path lines
    title : Optional[str], default=None
        Custom title for the plot
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure object
        
    Example:
    --------
    >>> # Get walk history from solver
    >>> solution, history = solver.solve(solve_points, nWalks=10, return_history=True)
    >>> 
    >>> # Plot first walk for first solve point
    >>> fig = plot_walk_history(history, point_idx=0, walk_idx=0)
    >>> plt.show()
    >>> 
    >>> # Plot with boundary visualization
    >>> polylines = {'dirichlet': dirichlet_boundary, 'neumann': neumann_boundary}
    >>> fig = plot_walk_history(history, point_idx=0, walk_idx=0, polylines=polylines)
    >>> plt.show()
    """
    
    # Validate inputs
    if point_idx not in walk_history:
        raise ValueError(f"Point index {point_idx} not found in walk history")
    
    point_walks = walk_history[point_idx]
    if walk_idx >= len(point_walks):
        raise ValueError(f"Walk index {walk_idx} not found for point {point_idx}")
    
    walk_data = point_walks[walk_idx]
    
    # Extract walk path and contributions
    path = walk_data['path']
    contributions = walk_data.get('contributions', [])
    walk_id = walk_data['walk_id']
    total_contribution = walk_data['total_contribution']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract path coordinates
    path_points = torch.stack([step['point'] for step in path])
    x_coords = path_points[:, 0].numpy()
    y_coords = path_points[:, 1].numpy()
    
    # Plot boundaries if provided
    if polylines is not None:
        if 'dirichlet' in polylines and polylines['dirichlet'] is not None:
            dirichlet_pts = polylines['dirichlet'].points.numpy()
            ax.plot(dirichlet_pts[:, 0], dirichlet_pts[:, 1], 
                   'r-', linewidth=2, label='Dirichlet Boundary', zorder=1)
        
        if 'neumann' in polylines and polylines['neumann'] is not None:
            neumann_pts = polylines['neumann'].points.numpy()
            ax.plot(neumann_pts[:, 0], neumann_pts[:, 1], 
                   'b-', linewidth=2, label='Neumann Boundary', zorder=1)
    
    # Plot step circles showing possible positions
    if show_step_circles:
        for i, step in enumerate(path[:-1]):  # Skip last point (boundary)
            point = step['point']
            d_dirichlet = step['dirichlet_distance']
            d_neumann = step.get('neumann_distance')
            
            x, y = point[0].item(), point[1].item()
            
            # Dirichlet circle (red)
            if d_dirichlet > 0:
                circle_d = Circle((x, y), d_dirichlet, fill=False, 
                                color='red', alpha=step_circle_alpha, 
                                linestyle='--', linewidth=1, zorder=2)
                ax.add_patch(circle_d)
            
            # Neumann circle (blue) - only if Neumann boundary exists
            if d_neumann is not None and d_neumann > 0:
                circle_n = Circle((x, y), d_neumann, fill=False, 
                                color='blue', alpha=step_circle_alpha, 
                                linestyle=':', linewidth=1, zorder=2)
                ax.add_patch(circle_n)
    
    # Plot path line connecting all points
    if show_path_line:
        ax.plot(x_coords, y_coords, 'k-', alpha=path_line_alpha, 
               linewidth=1.5, label='Walk Path', zorder=3)
    
    # Plot walk points
    ax.scatter(x_coords[:-1], y_coords[:-1], c='green', s=50, 
              alpha=0.7, edgecolors='black', linewidth=0.5, 
              label='Walk Points', zorder=4)
    
    # Highlight start and end points
    ax.scatter(x_coords[0], y_coords[0], c='lime', s=100, 
              marker='o', edgecolors='black', linewidth=2, 
              label='Start', zorder=5)
    ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100, 
              marker='X', edgecolors='black', linewidth=2, 
              label='End (Boundary)', zorder=5)
    
    # Plot contribution points
    if show_contributions and contributions:
        source_points = []
        boundary_points = []
        
        for contrib in contributions:
            point = contrib['point']
            contrib_type = contrib['type']
            
            if contrib_type == 'source':
                source_points.append(point)
            elif contrib_type == 'boundary':
                boundary_points.append(point)
        
        # Plot source contributions
        if source_points:
            source_array = torch.stack(source_points)
            ax.scatter(source_array[:, 0].numpy(), source_array[:, 1].numpy(), 
                      c='orange', s=80, marker='*', edgecolors='black', 
                      linewidth=1, label='Source Contributions', zorder=6)
        
        # Plot boundary contributions
        if boundary_points:
            boundary_array = torch.stack(boundary_points)
            ax.scatter(boundary_array[:, 0].numpy(), boundary_array[:, 1].numpy(), 
                      c='purple', s=80, marker='s', edgecolors='black', 
                      linewidth=1, label='Boundary Contributions', zorder=6)
    
    # Add step circles to legend if shown
    if show_step_circles:
        ax.plot([], [], 'r--', alpha=step_circle_alpha, 
               label='Dirichlet Distance')
        if any(step.get('neumann_distance') is not None for step in path):
            ax.plot([], [], 'b:', alpha=step_circle_alpha, 
                   label='Neumann Distance')
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Title
    if title is None:
        title = (f'Walk History - Point {point_idx}, Walk {walk_id}\n'
                f'Steps: {len(path)}, Total Contribution: {total_contribution:.6f}')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    
    return fig


def plot_multiple_walks(walk_history: Dict[int, List[Dict]], 
                       point_idx: int = 0, 
                       n_walks: int = 5,
                       polylines: Optional[Dict[str, Any]] = None,
                       figsize: Tuple[int, int] = (15, 10),
                       show_step_circles: bool = False,
                       alpha: float = 0.6) -> plt.Figure:
    """
    Plot multiple walks for the same solve point to visualize walk variability.
    
    Parameters:
    -----------
    walk_history : Dict[int, List[Dict]]
        Walk history dictionary from solver
    point_idx : int, default=0
        Index of the solve point to visualize
    n_walks : int, default=5
        Number of walks to plot
    polylines : Optional[Dict[str, Any]], default=None
        Dictionary containing boundary polylines objects
    figsize : Tuple[int, int], default=(15, 10)
        Figure size in inches
    show_step_circles : bool, default=False
        Whether to show step circles (can be cluttered with multiple walks)
    alpha : float, default=0.6
        Transparency of walk paths
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure object
    """
    
    if point_idx not in walk_history:
        raise ValueError(f"Point index {point_idx} not found in walk history")
    
    point_walks = walk_history[point_idx]
    n_walks = min(n_walks, len(point_walks))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot boundaries if provided
    if polylines is not None:
        if 'dirichlet' in polylines and polylines['dirichlet'] is not None:
            dirichlet_pts = polylines['dirichlet'].points.numpy()
            ax.plot(dirichlet_pts[:, 0], dirichlet_pts[:, 1], 
                   'r-', linewidth=3, label='Dirichlet Boundary', zorder=1)
        
        if 'neumann' in polylines and polylines['neumann'] is not None:
            neumann_pts = polylines['neumann'].points.numpy()
            ax.plot(neumann_pts[:, 0], neumann_pts[:, 1], 
                   'b-', linewidth=3, label='Neumann Boundary', zorder=1)
    
    # Color map for different walks
    colors = plt.cm.tab10(np.linspace(0, 1, n_walks))
    
    all_start_points = []
    all_end_points = []
    
    for i in range(n_walks):
        walk_data = point_walks[i]
        path = walk_data['path']
        walk_id = walk_data['walk_id']
        
        # Extract coordinates
        path_points = torch.stack([step['point'] for step in path])
        x_coords = path_points[:, 0].numpy()
        y_coords = path_points[:, 1].numpy()
        
        # Plot walk path
        ax.plot(x_coords, y_coords, color=colors[i], alpha=alpha, 
               linewidth=2, label=f'Walk {walk_id}', zorder=2)
        
        # Plot walk points
        ax.scatter(x_coords[:-1], y_coords[:-1], c=colors[i], s=30, 
                  alpha=alpha, edgecolors='black', linewidth=0.5, zorder=3)
        
        # Store start and end points
        all_start_points.append(path_points[0])
        all_end_points.append(path_points[-1])
        
        # Plot step circles for first walk only if requested
        if show_step_circles and i == 0:
            for step in path[:-1]:
                point = step['point']
                d_dirichlet = step['dirichlet_distance']
                d_neumann = step.get('neumann_distance')
                
                x, y = point[0].item(), point[1].item()
                
                if d_dirichlet > 0:
                    circle_d = Circle((x, y), d_dirichlet, fill=False, 
                                    color='red', alpha=0.3, 
                                    linestyle='--', linewidth=1, zorder=2)
                    ax.add_patch(circle_d)
                
                if d_neumann is not None and d_neumann > 0:
                    circle_n = Circle((x, y), d_neumann, fill=False, 
                                    color='blue', alpha=0.3, 
                                    linestyle=':', linewidth=1, zorder=2)
                    ax.add_patch(circle_n)
    
    # Plot common start and end points
    start_points = torch.stack(all_start_points)
    end_points = torch.stack(all_end_points)
    
    ax.scatter(start_points[:, 0].numpy(), start_points[:, 1].numpy(), 
              c='lime', s=150, marker='o', edgecolors='black', 
              linewidth=2, label='Start Points', zorder=4)
    ax.scatter(end_points[:, 0].numpy(), end_points[:, 1].numpy(), 
              c='red', s=150, marker='X', edgecolors='black', 
              linewidth=2, label='End Points', zorder=4)
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_title(f'Multiple Walks for Point {point_idx}\n'
                f'Showing {n_walks} walks', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    
    return fig


def plot_walk_statistics(walk_history: Dict[int, List[Dict]], 
                        point_idx: int = 0,
                        figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot statistics about walks for a given solve point.
    
    Parameters:
    -----------
    walk_history : Dict[int, List[Dict]]
        Walk history dictionary from solver
    point_idx : int, default=0
        Index of the solve point to analyze
    figsize : Tuple[int, int], default=(15, 5)
        Figure size in inches
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure object
    """
    
    if point_idx not in walk_history:
        raise ValueError(f"Point index {point_idx} not found in walk history")
    
    point_walks = walk_history[point_idx]
    
    # Extract statistics
    walk_lengths = [len(walk['path']) for walk in point_walks]
    contributions = [walk['total_contribution'] for walk in point_walks]
    
    # Count contribution types
    source_counts = []
    boundary_counts = []
    
    for walk in point_walks:
        source_count = sum(1 for c in walk.get('contributions', []) if c['type'] == 'source')
        boundary_count = sum(1 for c in walk.get('contributions', []) if c['type'] == 'boundary')
        source_counts.append(source_count)
        boundary_counts.append(boundary_count)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Walk lengths
    ax1 = axes[0]
    ax1.hist(walk_lengths, bins=min(20, len(set(walk_lengths))), 
             alpha=0.7, edgecolor='black')
    ax1.set_title('Walk Lengths')
    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Contributions
    ax2 = axes[1]
    ax2.hist(contributions, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_title('Total Contributions')
    ax2.set_xlabel('Contribution Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Contribution types
    ax3 = axes[2]
    width = 0.35
    x = np.arange(len(point_walks))
    
    ax3.bar(x - width/2, source_counts, width, label='Source', alpha=0.7)
    ax3.bar(x + width/2, boundary_counts, width, label='Boundary', alpha=0.7)
    ax3.set_title('Contribution Types by Walk')
    ax3.set_xlabel('Walk Index')
    ax3.set_ylabel('Number of Contributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Walk Statistics for Point {point_idx}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig