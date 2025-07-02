import torch
import numpy as np
import matplotlib.pyplot as plt
from WoStSolver import WostSolver_2D
from Polylines import PolyLines, PolyLinesSimple

def create_test_domain():
    """
    Create a test domain with mixed boundary conditions.
    - Outer boundary: square domain with Dirichlet conditions
    - Inner boundary: circular obstacle with Neumann conditions
    """
    
    # Create outer square boundary (Dirichlet)
    # Square from (-2, -2) to (2, 2)
    square_points = torch.tensor([
        [-2.0, -2.0],
        [2.0, -2.0], 
        [2.0, 2.0],
        [-2.0, 2.0],
        [-2.0, -2.0]  # Close the loop
    ])
    
    # Create inner circular boundary (Neumann)
    # Circle centered at origin with radius 0.5
    n_circle_points = 32
    theta = torch.linspace(0, 2*np.pi, n_circle_points + 1)
    radius = 0.5
    circle_points = torch.stack([
        radius * torch.cos(theta),
        radius * torch.sin(theta)
    ], dim=1)
    
    # Create polyline boundaries
    dirichlet_boundary = PolyLinesSimple(square_points)
    neumann_boundary = PolyLinesSimple(circle_points)
    
    return dirichlet_boundary, neumann_boundary

def define_boundary_conditions():
    """Define boundary conditions and source term"""
    
    # Dirichlet boundary condition: u = x^2 + y^2 on outer boundary
    def dirichlet_bc(point):
        x, y = point[0], point[1]
        return float(x**2 + y**2)
    
    # Source term: f(x,y) = -4 (so that u = x^2 + y^2 is the exact solution)
    def source_term(point):
        # check if the point is within problem bounds
        if point[0] < -2.0 or point[0] > 2.0 or point[1] < -2.0 or point[1] > 2.0:
            return 0.0
        return -4.0
    
    return dirichlet_bc, source_term

def create_solve_points():
    """Create a grid of points where we want to solve"""
    
    # Create a 21x21 grid from -1.8 to 1.8 (avoiding the inner circle)
    x = torch.linspace(-1.8, 1.8, 21)
    y = torch.linspace(-1.8, 1.8, 21)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten and combine
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Remove points inside the circle (radius 0.5)
    distances_from_origin = torch.norm(points, dim=1)
    valid_points = points[distances_from_origin > 0.6]  # Leave some margin
    
    return valid_points, X, Y

def analytical_solution(points):
    """Compute the analytical solution u = x^2 + y^2"""
    x, y = points[:, 0], points[:, 1]
    return x**2 + y**2

def run_test():
    """Run the complete test case"""
    
    print("Setting up test domain...")
    
    # Create domain
    dirichlet_boundary, neumann_boundary = create_test_domain()
    
    # Create solve points
    solve_points, X, Y = create_solve_points()
    
    # Define boundary conditions
    dirichlet_bc, source_term = define_boundary_conditions()
    
    print(f"Solving at {len(solve_points)} points...")
    
    # Initialize solver
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        neumannBoundary=neumann_boundary,
    )
    
    # Set boundary conditions and source term
    solver.setBoundaryConditions(dirichlet_bc)
    solver.setSourceTerm(source_term)
    
    # Solve with fewer walks for faster testing
    solution = solver.solve(solve_points, nWalks=100, maxSteps=500)
    
    # Compute analytical solution for comparison
    analytical = analytical_solution(solve_points)
    
    print("Plotting results...")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Numerical solution
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(solve_points[:, 0], solve_points[:, 1], 
                          c=solution.flatten(), s=50, cmap='viridis')
    ax1.set_title('Numerical Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1)
    
    # Add boundary visualization
    square_points = torch.tensor([[-2,-2],[2,-2],[2,2],[-2,2],[-2,-2]])
    ax1.plot(square_points[:, 0], square_points[:, 1], 'r-', linewidth=2, label='Dirichlet BC')
    
    theta = torch.linspace(0, 2*np.pi, 100)
    circle_x = 0.5 * torch.cos(theta)
    circle_y = 0.5 * torch.sin(theta)
    ax1.plot(circle_x, circle_y, 'b-', linewidth=2, label='Neumann BC')
    ax1.legend()
    
    # Plot 2: Analytical solution
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(solve_points[:, 0], solve_points[:, 1], 
                          c=analytical, s=50, cmap='viridis')
    ax2.set_title('Analytical Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2)
    
    # Plot 3: Error
    ax3 = axes[1, 0]
    error = torch.abs(solution.flatten() - analytical)
    scatter3 = ax3.scatter(solve_points[:, 0], solve_points[:, 1], 
                          c=error, s=50, cmap='Reds')
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3)
    
    # Plot 4: Error statistics
    ax4 = axes[1, 1]
    ax4.hist(error.numpy(), bins=20, alpha=0.7, edgecolor='black')
    ax4.set_title('Error Distribution')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    mean_error = torch.mean(error)
    max_error = torch.max(error)
    rmse = torch.sqrt(torch.mean(error**2))
    
    print(f"\nError Statistics:")
    print(f"Mean absolute error: {mean_error:.4f}")
    print(f"Maximum error: {max_error:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Relative RMSE: {rmse/torch.mean(analytical):.2%}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_test()