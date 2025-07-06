import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLines, PolyLinesSimple

def create_square_domain(domain_size=2.0):
    """Create a square domain with Dirichlet boundary conditions"""
    half_size = domain_size / 2.0
    square_points = torch.tensor([
        [-half_size, -half_size],
        [half_size, -half_size], 
        [half_size, half_size],
        [-half_size, half_size],
        [-half_size, -half_size]  # Close the loop
    ])
    return PolyLinesSimple(square_points)

def manufactured_solution_with_trig_function():
    """
    Manufactured solution with spatially varying diffusion and absorption:
    u(x,y) = sin(πx) * sin(πy)
    D(x,y) = 2 + x        (variable diffusion)
    σ(x,y) = y² + 1       (variable absorption)

    PDE: -∇·(D(x,y)∇u) + σ(x,y)u = f(x,y)
    """

    def analytical_solution(points):
        x, y = points[:, 0], points[:, 1]
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    def diffusion_coefficient(point):
        """D(x,y) = 2 + x"""
        return point[0] + 2

    def absorption_coefficient(point):
        """σ(x,y) = y² + 1"""
        return point[1]**2 + 1
    def boundary_condition(point):
        x, y = point[0], point[1]
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    def source_term(point):
        """
        Let u(x,y) = sin(πx) * sin(πy)
        Then:
        ∇u = [π cos(πx) sin(πy), π sin(πx) cos(πy)]
        ∇²u = -π² sin(πx) sin(πy) - π² sin(πx) sin(πy) = -2π² u

        D(x,y) = 2 + x ⇒ ∇D = [1, 0]
        ∇·(D∇u) = D ∇²u + ∇D · ∇u
                = (2 + x)(-2π² u) + 1 * π cos(πx) sin(πy) + 0
                = -2π²(2 + x) * u + π cos(πx) sin(πy)

        σ(x,y) = y² + 1

        So:
        f = -∇·(D∇u) + σu
          = 2π²(2 + x) * u - π cos(πx) sin(πy) + (y² + 1) * u
        """
        x, y = point[0], point[1]
        pi = torch.pi
        sin_px = torch.sin(pi * x)
        sin_py = torch.sin(pi * y)
        cos_px = torch.cos(pi * x)

        u = sin_px * sin_py
        diffusion_term = 2 * pi**2 * (2 + x) * u
        gradD_dot_gradu = pi * cos_px * sin_py
        absorption_term = (y**2 + 1) * u

        return (diffusion_term + gradD_dot_gradu - absorption_term)

    return analytical_solution, diffusion_coefficient, absorption_coefficient, boundary_condition, source_term


def manufactured_solution_with_polynomial():
    """
    Polynomial manufactured solution:
    u(x,y) = (1 - x²)(1 - y²)   (zero on boundary of [-1,1]×[-1,1])
    D(x,y) = 2 + 0.5*x + 0.5*y  (linear diffusion)
    α(x,y) = 2 + xy             (bilinear absorption)

    PDE: -∇·(D(x,y)∇u) + α(x,y)u = f(x,y)
    """

    def analytical_solution(points):
        x, y = points[:, 0], points[:, 1]
        return (1 - x**2) * (1 - y**2)

    def diffusion_coefficient(point):
        """D(x,y) = 2 + 0.5*x + 0.5*y"""
        return 2.0 + 0.5 * point[0] + 0.5 * point[1]
    def absorption_coefficient(point):
        """α(x,y) = 2 + xy"""
        return point[0] * point[1] + 2

    def boundary_condition(point):
        x, y = point[0], point[1]
        return (1 - x**2) * (1 - y**2)

    def source_term(point):
        """
        Let u(x,y) = (1 - x²)(1 - y²)
        Then:
        ∇u = [-2x(1 - y²), -2y(1 - x²)]
        
        ∂²u/∂x² = -2(1 - y²)
        ∂²u/∂y² = -2(1 - x²)
        ∇²u = -2(1 - y²) - 2(1 - x²) = -2(2 - x² - y²)

        D(x,y) = 2 + 0.5*x + 0.5*y ⇒ ∇D = [0.5, 0.5]
        ∇·(D∇u) = D ∇²u + ∇D · ∇u
                = (2 + 0.5*x + 0.5*y)*(-2(2 - x² - y²)) + 0.5*(-2x(1 - y²)) + 0.5*(-2y(1 - x²))
                = -2(2 + 0.5*x + 0.5*y)(2 - x² - y²) - x(1 - y²) - y(1 - x²)

        α(x,y) = 2 + xy

        So:
        f = -∇·(D∇u) + αu
          = 2(2 + 0.5*x + 0.5*y)(2 - x² - y²) + x(1 - y²) + y(1 - x²) + (2 + xy)(1 - x²)(1 - y²)
        """
        x, y = point[0], point[1]
        u = (1 - x**2) * (1 - y**2)
        
        # Gradient and Laplacian terms
        laplacian_u = -2 * (2 - x**2 - y**2)
        D = 2 + 0.5*x + 0.5*y
        gradD_dot_gradu = -x*(1 - y**2) - y*(1 - x**2)
        div_D_grad_u = D * laplacian_u + gradD_dot_gradu
        
        # Absorption term
        alpha = 2 + x * y
        absorption_term = alpha * u
        
        return -div_D_grad_u + absorption_term

    return analytical_solution, diffusion_coefficient, absorption_coefficient, boundary_condition, source_term

def create_test_points(domain_size=2.0, n_points=4):
    """Create a small grid of test points"""
    half_size = domain_size / 2.0
    margin = 0.3
    coord_range = half_size - margin
    
    # Create a small n_points x n_points grid
    x = torch.linspace(-coord_range, coord_range, n_points)
    y = torch.linspace(-coord_range, coord_range, n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    return points


def run_test():
    """Run the manufactured solution test with variable coefficients"""
    
    # Create domain and test points
    boundary = create_square_domain(domain_size=2.0)
    test_points = create_test_points(domain_size=2.0, n_points=10)  # Only 16 points
    
    print(f"\nTesting at {len(test_points)} points")
    
    # Get manufactured solution components
    (analytical_solution, diffusion_coefficient, absorption_coefficient, 
     boundary_condition, source_term) = manufactured_solution_with_polynomial()
    
    # Initialize solver with variable coefficients
    solver = WostSolver_2D(
        dirichletBoundary=boundary,
        neumannBoundary=None,
        source=source_term,
        alpha=diffusion_coefficient,    # Pass the diffusion function
        sigma=absorption_coefficient   # Pass the absorption function
    )
    
    solver.setBoundaryConditions(boundary_condition)
    solver.setSourceTerm(source_term)
    
    # Solve with moderate number of walks for speed
    print("Solving (this may take a moment)...")
    solution = solver.solve(test_points, nWalks=125, maxSteps=800)
    
    # Compute analytical solution and errors
    analytical = analytical_solution(test_points)
    error = torch.abs(solution.flatten() - analytical).detach()
    relative_error = error / (torch.abs(analytical) + 1e-10)
    
    # Print results
    print(f"\nResults:")
    print(f"Solution range: [{torch.min(analytical):.3f}, {torch.max(analytical):.3f}]")
    print(f"Mean absolute error: {torch.mean(error):.4f}")
    print(f"Max absolute error: {torch.max(error):.4f}")
    print(f"RMSE: {torch.sqrt(torch.mean(error**2)):.4f}")
    print(f"Mean relative error: {torch.mean(relative_error):.2%}")
    print(f"Max relative error: {torch.max(relative_error):.2%}")
    
    # Create detailed comparison table
    print(f"\nPoint-by-point comparison:")
    print(f"{'x':>6} {'y':>6} {'Analytical':>12} {'Numerical':>12} {'Error':>10} {'Rel.Err':>8}")
    print("-" * 60)
    for i in range(len(test_points)):
        x, y = test_points[i, 0].item(), test_points[i, 1].item()
        anal = analytical[i].item()
        numer = solution.flatten()[i].item()
        err = error[i].item()
        rel_err = relative_error[i].item()
        print(f"{x:6.2f} {y:6.2f} {anal:12.4f} {numer:12.4f} {err:10.4f} {rel_err:7.1%}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    test_points_d = test_points.detach().numpy()
    # Plot numerical solution
    scatter1 = ax1.scatter(test_points_d[:, 0], test_points_d[:, 1], c=solution.detach().flatten(), 
                          s=100, cmap='viridis')
    ax1.set_title('Numerical Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1)
    
    # Plot analytical solution
    scatter2 = ax2.scatter(test_points_d[:, 0], test_points_d[:, 1], c=analytical, 
                          s=100, cmap='viridis')
    ax2.set_title('Analytical Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2)
    
    # Plot absolute error
    scatter3 = ax3.scatter(test_points_d[:, 0], test_points_d[:, 1], c=error, 
                          s=100, cmap='Reds')
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3)
    
    solution = solution.detach()
    # Plot numerical vs analytical
    ax4.scatter(analytical.numpy(), solution.flatten().numpy(), s=100, alpha=0.7)
    min_val = min(torch.min(analytical), torch.min(solution))
    max_val = max(torch.max(analytical), torch.max(solution))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Agreement')
    ax4.set_xlabel('Analytical Solution')
    ax4.set_ylabel('Numerical Solution')
    ax4.set_title('Solution Comparison')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    return solution, analytical, error

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_test()