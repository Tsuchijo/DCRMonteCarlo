import torch
import pytest
import numpy as np

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