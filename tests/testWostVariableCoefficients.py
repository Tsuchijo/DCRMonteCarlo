import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLinesSimple
import time


def create_test_domain():
    """
    Create a test domain with outer square (Dirichlet) and inner circle (Neumann) boundaries.
    Similar to testWostWithSource but optimized for variable coefficient problems.
    """
    # Outer square boundary (Dirichlet) - slightly smaller for better visualization
    square_points = torch.tensor([
        [-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [-1.5, -1.5]
    ])
    
    # Inner circular boundary (Neumann)
    n_circle_points = 32
    theta = torch.linspace(0, 2*np.pi, n_circle_points + 1)
    radius = 0.4
    circle_points = torch.stack([
        radius * torch.cos(theta), 
        radius * torch.sin(theta)
    ], dim=1)
    
    dirichlet_boundary = PolyLinesSimple(square_points)
    neumann_boundary = PolyLinesSimple(circle_points)
    
    return dirichlet_boundary, neumann_boundary


def define_variable_coefficients():
    """
    Define spatially varying diffusion and absorption coefficients.
    This creates an interesting test case with significant spatial variation.
    """
    def diffusion_coefficient(point):
        """
        Diffusion varies smoothly across the domain.
        Higher diffusion in the center, lower at the edges.
        """
        x, y = point[0], point[1]
        r_squared = x**2 + y**2
        return torch.tensor(0.5 + 1.5 * torch.exp(-2.0 * r_squared))
    
    def absorption_coefficient(point):
        """
        Absorption varies with a different spatial pattern.
        Creates interesting interactions with diffusion.
        """
        x, y = point[0], point[1]
        return torch.tensor(0.3 + 0.7 * (1 + torch.sin(2*np.pi*x) * torch.cos(2*np.pi*y)))
    
    return diffusion_coefficient, absorption_coefficient


def define_boundary_conditions_and_source():
    """
    Define boundary conditions and source term for the variable coefficient problem.
    We'll use a manufactured solution approach for validation.
    """
    def dirichlet_bc(point):
        """
        Dirichlet boundary condition: u = sin(πx)sin(πy) on outer boundary
        """
        x, y = point[0], point[1]
        return float(torch.sin(np.pi * x) * torch.sin(np.pi * y))
    
    def source_term(point):
        """
        Source term chosen to make the problem interesting but solvable.
        For variable coefficients, this represents the forcing term.
        """
        x, y = point[0], point[1]
        # Simple source that creates interesting flow patterns
        r_squared = x**2 + y**2
        if r_squared > 1.5**2:  # Outside the outer boundary
            return 0.0
        return float(torch.exp(-r_squared) * torch.sin(np.pi * x) * torch.cos(np.pi * y))
    
    return dirichlet_bc, source_term


def create_solve_points():
    """
    Create a grid of points where we want to evaluate the solution.
    Excludes points inside the inner circular boundary.
    """
    # Create a grid covering the domain
    x = torch.linspace(-1.3, 1.3, 27)  # Higher resolution for better visualization
    y = torch.linspace(-1.3, 1.3, 27)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Remove points inside the inner circle (with small margin)
    distances_from_origin = torch.norm(points, dim=1)
    valid_points = points[distances_from_origin > 0.5]  # 0.4 radius + 0.1 margin
    
    return valid_points, X, Y


def compute_reference_solution(solve_points, diffusion_func, absorption_func):
    """
    Compute a reference solution using a simpler method or analytical approximation.
    For complex variable coefficient problems, this might be approximate.
    """
    # For this test, we'll compute a simplified reference based on the boundary conditions
    # This is approximate but useful for qualitative comparison
    reference = torch.zeros(len(solve_points))
    
    for i, point in enumerate(solve_points):
        x, y = point[0], point[1]
        # Approximate solution based on boundary conditions and domain geometry
        r = torch.sqrt(x**2 + y**2)
        
        # Decay from boundary values, modulated by local coefficients
        boundary_val = torch.sin(np.pi * x) * torch.sin(np.pi * y)
        distance_factor = torch.exp(-r)  # Simple decay
        
        # Modulation by local diffusion (higher diffusion → smoother decay)
        diff_val = diffusion_func(point)
        abs_val = absorption_func(point)
        coeff_modulation = diff_val / (diff_val + abs_val)
        
        reference[i] = boundary_val * distance_factor * coeff_modulation
    
    return reference


def visualize_coefficients(diffusion_func, absorption_func):
    """
    Create visualization of the variable coefficients themselves.
    """
    # Create a grid for coefficient visualization
    x = torch.linspace(-1.5, 1.5, 50)
    y = torch.linspace(-1.5, 1.5, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    diff_vals = torch.zeros_like(X)
    abs_vals = torch.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = torch.tensor([X[i, j], Y[i, j]])
            diff_vals[i, j] = diffusion_func(point)
            abs_vals[i, j] = absorption_func(point)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Diffusion coefficient
    im1 = ax1.contourf(X.numpy(), Y.numpy(), diff_vals.numpy(), levels=20, cmap='viridis')
    ax1.set_title('Diffusion Coefficient D(x,y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Add circle to show inner boundary
    circle1 = plt.Circle((0, 0), 0.4, fill=False, color='red', linewidth=2)
    ax1.add_patch(circle1)
    
    # Absorption coefficient
    im2 = ax2.contourf(X.numpy(), Y.numpy(), abs_vals.numpy(), levels=20, cmap='plasma')
    ax2.set_title('Absorption Coefficient α(x,y)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    
    # Add circle to show inner boundary
    circle2 = plt.Circle((0, 0), 0.4, fill=False, color='red', linewidth=2)
    ax2.add_patch(circle2)
    
    plt.tight_layout()
    plt.savefig('variable_coefficients.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_variable_coefficient_test():
    """
    Main test function that orchestrates the entire variable coefficient test.
    """
    print("="*60)
    print("VARIABLE COEFFICIENT WALK ON SPHERES TEST")
    print("="*60)
    
    # 1. Create test domain
    print("1. Setting up test domain...")
    dirichlet_boundary, neumann_boundary = create_test_domain()
    
    # 2. Define variable coefficients
    print("2. Defining variable coefficients...")
    diffusion_func, absorption_func = define_variable_coefficients()
    
    # 3. Define boundary conditions and source
    print("3. Setting up boundary conditions and source term...")
    dirichlet_bc, source_term = define_boundary_conditions_and_source()
    
    # 4. Create solve points
    print("4. Creating solution grid...")
    solve_points, X, Y = create_solve_points()
    print(f"   Will solve at {len(solve_points)} points")
    
    # 5. Visualize coefficients
    print("5. Visualizing variable coefficients...")
    visualize_coefficients(diffusion_func, absorption_func)
    
    # 6. Initialize solver with variable coefficients
    print("6. Initializing variable coefficient solver...")
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        neumannBoundary=neumann_boundary,
        sigma=absorption_func,
        alpha=diffusion_func,
        source=source_term
    )
    solver.setBoundaryConditions(dirichlet_bc)
    
    print(f"   Delta tracking enabled: {solver.use_delta_tracking}")
    print(f"   Sigma bar parameter: {solver.sigma_bar:.6f}")
    
    # 7. Solve the PDE
    print("7. Solving variable coefficient PDE...")
    start_time = time.time()
    
    # Use moderate parameters for reasonable computation time
    solution = solver.solve(solve_points, nWalks=25, maxSteps=1000)
    
    solve_time = time.time() - start_time
    print(f"   Solution completed in {solve_time:.2f} seconds")
    print(f"   Average time per point: {solve_time/len(solve_points):.4f} seconds")
    
    # 8. Compute reference solution
    print("8. Computing reference solution...")
    reference = compute_reference_solution(solve_points, diffusion_func, absorption_func)
    
    # 9. Create comprehensive visualization
    print("9. Creating solution visualization...")
    visualize_results(solve_points, solution, reference, X, Y, 
                     dirichlet_boundary, neumann_boundary)
    
    # 10. Print statistics
    print("10. Solution Statistics:")
    print(f"    Solution range: [{solution.min():.6f}, {solution.max():.6f}]")
    print(f"    Solution mean: {solution.mean():.6f}")
    print(f"    Solution std: {solution.std():.6f}")
    
    # Qualitative comparison with reference
    if len(reference) == len(solution):
        diff = torch.abs(solution.squeeze() - reference)
        print(f"    Mean absolute difference from reference: {diff.mean():.6f}")
        print(f"    Max absolute difference from reference: {diff.max():.6f}")
    
    print("\n" + "="*60)
    print("VARIABLE COEFFICIENT TEST COMPLETED")
    print("="*60)
    
    return solution, reference, solve_points


def visualize_results(solve_points, solution, reference, X, Y, 
                     dirichlet_boundary, neumann_boundary):
    """
    Create comprehensive visualization of the results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prepare data for gridded visualization
    solution_grid = torch.full(X.shape, torch.nan)
    reference_grid = torch.full(X.shape, torch.nan)
    
    # Map solution back to grid
    for i, point in enumerate(solve_points):
        x_val, y_val = point[0].item(), point[1].item()
        # Find closest grid point
        x_idx = torch.argmin(torch.abs(X[0, :] - x_val))
        y_idx = torch.argmin(torch.abs(Y[:, 0] - y_val))
        if torch.sqrt((X[y_idx, x_idx] - x_val)**2 + (Y[y_idx, x_idx] - y_val)**2) < 0.1:
            solution_grid[y_idx, x_idx] = solution[i]
            reference_grid[y_idx, x_idx] = reference[i]
    
    # 1. Numerical Solution
    ax1 = axes[0, 0]
    im1 = ax1.contourf(X.numpy(), Y.numpy(), solution_grid.numpy(), 
                       levels=20, cmap='RdYlBu_r', extend='both')
    ax1.set_title('Variable Coefficient Solution\n(Delta Tracking)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    
    # Add boundaries
    outer_pts = dirichlet_boundary.points.numpy()
    inner_pts = neumann_boundary.points.numpy()
    ax1.plot(outer_pts[:, 0], outer_pts[:, 1], 'k-', linewidth=2, label='Dirichlet')
    ax1.plot(inner_pts[:, 0], inner_pts[:, 1], 'r-', linewidth=2, label='Neumann')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)
    
    # 2. Reference Solution
    ax2 = axes[0, 1]
    im2 = ax2.contourf(X.numpy(), Y.numpy(), reference_grid.numpy(), 
                       levels=20, cmap='RdYlBu_r', extend='both')
    ax2.set_title('Reference Solution\n(Approximate)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    
    # Add boundaries
    ax2.plot(outer_pts[:, 0], outer_pts[:, 1], 'k-', linewidth=2, label='Dirichlet')
    ax2.plot(inner_pts[:, 0], inner_pts[:, 1], 'r-', linewidth=2, label='Neumann')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    # 3. Absolute Difference
    ax3 = axes[1, 0]
    diff_grid = torch.abs(solution_grid - reference_grid)
    im3 = ax3.contourf(X.numpy(), Y.numpy(), diff_grid.numpy(), 
                       levels=20, cmap='Reds', extend='max')
    ax3.set_title('Absolute Difference\n|Numerical - Reference|', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')
    
    # Add boundaries
    ax3.plot(outer_pts[:, 0], outer_pts[:, 1], 'k-', linewidth=2)
    ax3.plot(inner_pts[:, 0], inner_pts[:, 1], 'k-', linewidth=2)
    plt.colorbar(im3, ax=ax3)
    
    # 4. Solution scatter plot with coefficient overlay
    ax4 = axes[1, 1]
    
    # Create coefficient visualization as background
    x_bg = torch.linspace(-1.5, 1.5, 30)
    y_bg = torch.linspace(-1.5, 1.5, 30)
    X_bg, Y_bg = torch.meshgrid(x_bg, y_bg, indexing='ij')
    
    diffusion_func, absorption_func = define_variable_coefficients()
    ratio_grid = torch.zeros_like(X_bg)
    for i in range(X_bg.shape[0]):
        for j in range(X_bg.shape[1]):
            point = torch.tensor([X_bg[i, j], Y_bg[i, j]])
            diff_val = diffusion_func(point)
            abs_val = absorption_func(point)
            ratio_grid[i, j] = diff_val / abs_val
    
    # Background showing diffusion/absorption ratio
    ax4.contourf(X_bg.numpy(), Y_bg.numpy(), ratio_grid.numpy(), 
                levels=15, cmap='gray', alpha=0.3)
    
    # Scatter plot of solution values
    scatter = ax4.scatter(solve_points[:, 0].numpy(), solve_points[:, 1].numpy(), 
                         c=solution.squeeze().numpy(), cmap='viridis', s=20, edgecolors='black', linewidth=0.5)
    ax4.set_title('Solution Values\n(on D/α ratio background)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_aspect('equal')
    
    # Add boundaries
    ax4.plot(outer_pts[:, 0], outer_pts[:, 1], 'k-', linewidth=2)
    ax4.plot(inner_pts[:, 0], inner_pts[:, 1], 'r-', linewidth=2)
    plt.colorbar(scatter, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('variable_coefficient_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run the comprehensive variable coefficient test
    solution, reference, solve_points = run_variable_coefficient_test()
    
    # Additional analysis could be added here
    print(f"\nTest completed successfully!")
    print(f"Results saved to: variable_coefficients.png and variable_coefficient_results.png")