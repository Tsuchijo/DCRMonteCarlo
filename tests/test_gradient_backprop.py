import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLinesSimple

# Try to import matplotlib, but don't fail if it's not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class ParametricAbsorption(nn.Module):
    """
    Parametric absorption function with learnable parameters.
    Models alpha(x,y) = base_alpha + amplitude * exp(-((x-center_x)^2 + (y-center_y)^2) / width^2)
    """
    
    def __init__(self, base_alpha=1.0, amplitude=0.5, center_x=0.0, center_y=0.0, width=0.5):
        super().__init__()
        # Make all parameters learnable
        self.base_alpha = nn.Parameter(torch.tensor(base_alpha, dtype=torch.float32))
        self.amplitude = nn.Parameter(torch.tensor(amplitude, dtype=torch.float32))
        self.center_x = nn.Parameter(torch.tensor(center_x, dtype=torch.float32))
        self.center_y = nn.Parameter(torch.tensor(center_y, dtype=torch.float32))
        self.width = nn.Parameter(torch.tensor(width, dtype=torch.float32))
    
    def forward(self, point):
        """
        Evaluate absorption function at given point.
        
        Args:
            point: torch.Tensor of shape (2,) representing (x, y)
        
        Returns:
            torch.Tensor: absorption value at the point
        """
        x, y = point[0], point[1]
        
        # Gaussian-like absorption variation
        distance_sq = (x - self.center_x)**2 + (y - self.center_y)**2
        gaussian = torch.exp(-distance_sq / (self.width**2 + 1e-8))  # Add small epsilon for stability
        
        alpha = self.base_alpha + self.amplitude * gaussian
        
        # Ensure alpha is positive
        return torch.clamp(alpha, min=0.01)
    
    def visualize(self, domain_bounds, resolution=50):
        """Visualize the absorption function over the domain."""
        x = torch.linspace(domain_bounds[0][0], domain_bounds[0][1], resolution)
        y = torch.linspace(domain_bounds[1][0], domain_bounds[1][1], resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        Z = torch.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                point = torch.tensor([X[i, j], Y[i, j]])
                Z[i, j] = self.forward(point)
        
        return X.detach().numpy(), Y.detach().numpy(), Z.detach().numpy()

def create_simple_test_domain():
    """Create a simple square domain for testing."""
    # Square domain from -1 to 1
    square_points = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]
    ])
    dirichlet_boundary = PolyLinesSimple(square_points)
    
    return dirichlet_boundary

def test_gradient_backpropagation():
    """
    Test gradient backpropagation from solver output to absorption function parameters.
    """
    print("="*60)
    print("GRADIENT BACKPROPAGATION TEST")
    print("="*60)
    
    # Create domain
    dirichlet_boundary = create_simple_test_domain()
    
    # Create parametric absorption function
    alpha_func = ParametricAbsorption(
        base_alpha=1.0,
        amplitude=0.5, 
        center_x=0.0,
        center_y=0.0,
        width=0.3
    )
    
    # Define simple diffusion (constant)
    def diffusion_func(point):
        return torch.tensor(1.0, dtype=torch.float32)
    
    # Define boundary condition
    def boundary_condition(point):
        return 0.0  # Zero Dirichlet BC
    
    # Define source term
    def source_func(point):
        return 1.0  # Constant source
    
    # Create test points
    test_points = torch.tensor([
        [0.2, 0.2],
        [-0.3, 0.4],
        [0.5, -0.2]
    ])
    
    # Target values (what we want the solution to be at test points)
    target_values = torch.tensor([0.5, 0.8, 0.3]).unsqueeze(1)
    
    print(f"Initial parameters:")
    print(f"  base_alpha: {alpha_func.base_alpha.item():.4f}")
    print(f"  amplitude: {alpha_func.amplitude.item():.4f}")
    print(f"  center: ({alpha_func.center_x.item():.4f}, {alpha_func.center_y.item():.4f})")
    print(f"  width: {alpha_func.width.item():.4f}")
    
    # Create solver with gradient-enabled absorption function
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        diffusion=diffusion_func,
        absorption=alpha_func.forward,  # Use the forward method
        source=source_func
    )
    solver.setBoundaryConditions(boundary_condition)
    
    print(f"\nSolver configuration:")
    print(f"  Delta tracking enabled: {solver.use_delta_tracking}")
    print(f"  Sigma bar: {solver.sigma_bar:.6f}")
    
    # Forward pass through solver
    print(f"\nRunning forward pass...")
    try:
        # Use fewer walks for testing, but enough to get meaningful gradients
        solution = solver.solve(test_points, nWalks=100, maxSteps=200)
        print(f"Solution shape: {solution.shape}")
        print(f"Solution values: {solution.flatten()}")
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(solution, target_values)
        print(f"Loss (MSE): {loss.item():.6f}")
        
        # Check if gradients are enabled
        print(f"\nGradient status:")
        print(f"  base_alpha.requires_grad: {alpha_func.base_alpha.requires_grad}")
        print(f"  amplitude.requires_grad: {alpha_func.amplitude.requires_grad}")
        print(f"  solution.requires_grad: {solution.requires_grad}")
        
        if solution.requires_grad:
            print(f"\n✅ Forward pass successful with gradients!")
            
            # Backward pass
            print(f"Running backward pass...")
            loss.backward()
            
            # Check gradients
            print(f"\nGradients:")
            print(f"  base_alpha.grad: {alpha_func.base_alpha.grad}")
            print(f"  amplitude.grad: {alpha_func.amplitude.grad}")
            print(f"  center_x.grad: {alpha_func.center_x.grad}")
            print(f"  center_y.grad: {alpha_func.center_y.grad}")
            print(f"  width.grad: {alpha_func.width.grad}")
            
            # Check if any gradients are non-zero
            has_gradients = any([
                alpha_func.base_alpha.grad is not None and alpha_func.base_alpha.grad.abs() > 1e-8,
                alpha_func.amplitude.grad is not None and alpha_func.amplitude.grad.abs() > 1e-8,
                alpha_func.center_x.grad is not None and alpha_func.center_x.grad.abs() > 1e-8,
                alpha_func.center_y.grad is not None and alpha_func.center_y.grad.abs() > 1e-8,
                alpha_func.width.grad is not None and alpha_func.width.grad.abs() > 1e-8,
            ])
            
            if has_gradients:
                print(f"\n🎉 SUCCESS: Gradients successfully backpropagated!")
                return True, loss.item(), alpha_func
            else:
                print(f"\n⚠️  WARNING: Gradients are zero or None")
                return False, loss.item(), alpha_func
        else:
            print(f"\n❌ ISSUE: Solution does not require gradients")
            return False, loss.item(), alpha_func
            
    except Exception as e:
        print(f"\n❌ ERROR during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False, float('inf'), alpha_func

def test_gradient_optimization():
    """
    Test a simple optimization loop using the gradients.
    """
    print(f"\n" + "="*60)
    print("GRADIENT OPTIMIZATION TEST")
    print("="*60)
    
    # Create domain and absorption function
    dirichlet_boundary = create_simple_test_domain()
    alpha_func = ParametricAbsorption(base_alpha=1.0, amplitude=0.3)
    
    # Simple diffusion and boundary conditions
    def diffusion_func(point):
        return torch.tensor(1.0, dtype=torch.float32)
    
    def boundary_condition(point):
        return 0.0
    
    def source_func(point):
        return 1.0
    
    # Test points and targets
    test_points = torch.tensor([[0.0, 0.0], [0.3, 0.3]])
    target_values = torch.tensor([0.6, 0.4]).unsqueeze(1)
    
    # Create optimizer
    optimizer = torch.optim.Adam(alpha_func.parameters(), lr=0.01)
    
    print(f"Running optimization for 5 steps...")
    
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        
        # Create fresh solver for each step
        solver = WostSolver_2D(
            dirichletBoundary=dirichlet_boundary,
            diffusion=diffusion_func,
            absorption=alpha_func.forward,
            source=source_func
        )
        solver.setBoundaryConditions(boundary_condition)
        
        try:
            # Forward pass
            solution = solver.solve(test_points, nWalks=50, maxSteps=100)
            loss = torch.nn.functional.mse_loss(solution, target_values)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"  Step {step+1}: loss = {loss.item():.6f}, base_alpha = {alpha_func.base_alpha.item():.4f}")
            
        except Exception as e:
            print(f"  Step {step+1}: ERROR - {e}")
            break
    
    print(f"\nOptimization complete!")
    print(f"Loss evolution: {losses}")
    
    return losses, alpha_func

def visualize_absorption_function(alpha_func, domain_bounds):
    """Visualize the absorption function."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping visualization")
        return
        
    X, Y, Z = alpha_func.visualize(domain_bounds)
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Absorption α(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Parametric Absorption Function')
    plt.axis('equal')
    
    # Mark the center
    plt.plot(alpha_func.center_x.item(), alpha_func.center_y.item(), 'r*', markersize=10, label='Center')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test gradient backpropagation
    success, loss, alpha_func = test_gradient_backpropagation()
    
    if success:
        print(f"\n🎉 Gradient backpropagation working!")
        
        # Visualize the absorption function
        domain_bounds = [[-1, 1], [-1, 1]]
        visualize_absorption_function(alpha_func, domain_bounds)
        
        # Test optimization
        losses, optimized_alpha = test_gradient_optimization()
        
        if losses and len(losses) > 1:
            print(f"\n📈 Optimization shows learning: {losses[0]:.6f} → {losses[-1]:.6f}")
        
    else:
        print(f"\n❌ Gradient backpropagation failed!")
        print(f"This suggests the solver is not differentiable end-to-end")
        print(f"Possible issues:")
        print(f"  - Non-differentiable operations in solver")
        print(f"  - Gradient flow blocked by detach() calls")
        print(f"  - Random sampling breaking gradient flow")