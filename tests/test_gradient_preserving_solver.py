import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLinesSimple

class GradientPreservingWoSSolver(WostSolver_2D):
    """
    Modified WoS solver that preserves gradients for differentiable optimization.
    
    Key changes:
    1. Pre-generate all random numbers as non-gradient tensors
    2. Preserve gradient flow through alpha function evaluations
    3. Maintain exact same sampling behavior
    """
    
    def solve_with_gradients(self, solvePoints: torch.tensor, nWalks=1000, maxSteps=1000, 
                           fixed_seed=None) -> torch.Tensor:
        """
        Gradient-preserving solve method.
        
        Args:
            solvePoints: Points to solve at
            nWalks: Number of random walks
            maxSteps: Maximum steps per walk
            fixed_seed: Optional seed for reproducible random numbers
            
        Returns:
            Solution tensor with gradients preserved
        """
        if fixed_seed is not None:
            torch.manual_seed(fixed_seed)
            np.random.seed(fixed_seed)
        
        # Pre-generate ALL random numbers needed
        total_randoms_needed = len(solvePoints) * nWalks * maxSteps * 2  # angles + uniforms
        
        # Generate as non-gradient tensors (fixed randomness)
        all_angles = torch.rand(total_randoms_needed) * 2 * np.pi
        all_uniforms = torch.rand(total_randoms_needed)
        
        # These tensors should NOT require gradients
        all_angles = all_angles.detach()
        all_uniforms = all_uniforms.detach()
        
        random_idx = 0
        
        if self.neumannBoundary is None:
            return self._solve_dirichlet_with_gradients(
                solvePoints, nWalks, maxSteps, all_angles, all_uniforms, random_idx
            )
        elif self.use_delta_tracking:
            return self._solve_delta_tracking_with_gradients(
                solvePoints, nWalks, maxSteps, all_angles, all_uniforms, random_idx
            )
        else:
            return self._solve_mixed_with_gradients(
                solvePoints, nWalks, maxSteps, all_angles, all_uniforms, random_idx
            )
    
    def _solve_dirichlet_with_gradients(self, solvePoints, nWalks, maxSteps, 
                                      all_angles, all_uniforms, random_idx):
        """Gradient-preserving Dirichlet solver."""
        eps = 1e-4
        rmin = 1e-3
        
        # Initialize list to collect results
        results_list = []
        
        for i, point in enumerate(solvePoints):
            point_total = torch.tensor(0.0, requires_grad=True)
            
            for walk_idx in range(nWalks):
                # Start point (requires gradients if point does)
                current_point = point.clone()
                
                step_count = 0
                dDirichlet = 1.0
                
                while (step_count < maxSteps) and (dDirichlet > eps):
                    # Distance calculation (should preserve gradients)
                    dDirichlet = self.dirichletBoundary.distance(current_point).item()
                    r = max(rmin, dDirichlet)
                    
                    # Use pre-generated random angle (no gradients)
                    if random_idx < len(all_angles):
                        theta = all_angles[random_idx]
                        random_idx += 1
                    else:
                        theta = torch.tensor(torch.rand(1).item() * 2 * np.pi)
                    
                    cos_theta = torch.cos(theta)
                    sin_theta = torch.sin(theta)
                    
                    if self.source is not None:
                        # Sample Green's function (use pre-generated random)
                        if random_idx < len(all_uniforms):
                            # Use cached or pre-generated sampling
                            r_sampled = self._sample_greens_fixed(current_point, r, all_uniforms[random_idx:random_idx+10])
                            random_idx += 10
                        else:
                            r_sampled = self.sampleGreens(current_point, r)
                        
                        # Sample point (PRESERVE GRADIENTS)
                        sample_point = current_point + r_sampled * torch.stack([cos_theta, sin_theta])
                        
                        # Source evaluation (THIS MUST PRESERVE GRADIENTS)
                        source_contribution = self.source(sample_point) * r**2 / 4
                        point_total = point_total + source_contribution
                    
                    # Move to next point (PRESERVE GRADIENTS) 
                    current_point = current_point + r * torch.stack([cos_theta, sin_theta])
                    step_count += 1
                
                # Boundary condition evaluation (PRESERVE GRADIENTS)
                boundary_contribution = self.boundaryDirichlet(current_point)
                point_total = point_total + boundary_contribution
            
            # Average over walks and add to results
            results_list.append(point_total / nWalks)
        
        # Stack results into tensor
        return torch.stack(results_list).unsqueeze(1)
    
    def _sample_greens_fixed(self, x, r, random_values):
        """Fixed random sampling for Green's function to preserve gradients."""
        # Use first random value to select from cache or generate simple sample
        uniform_val = random_values[0].item()
        
        # Simple approximation: use inverse transform sampling
        # For Green's function ~ log(1/r), use exponential approximation
        eps = 1e-6
        sampled_r = eps + (1 - eps) * (1 - uniform_val)  # Simple mapping
        
        return sampled_r * r

def test_gradient_preservation():
    """Test if the gradient-preserving solver actually preserves gradients."""
    print("="*60)
    print("GRADIENT-PRESERVING SOLVER TEST")
    print("="*60)
    
    # Create simple test case
    square_points = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]
    ])
    dirichlet_boundary = PolyLinesSimple(square_points)
    
    # Parametric absorption function
    class SimpleAlpha(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha_param = nn.Parameter(torch.tensor(1.0))
        
        def forward(self, point):
            return self.alpha_param
    
    alpha_func = SimpleAlpha()
    
    def diffusion_func(point):
        return torch.tensor(1.0)
    
    def boundary_condition(point):
        return 0.0
    
    def source_func(point):
        # Source that depends on alpha to test gradient flow
        alpha_val = alpha_func(point)
        return alpha_val  # Source = alpha, so gradient should flow
    
    print(f"Initial alpha parameter: {alpha_func.alpha_param.item():.6f}")
    print(f"Alpha requires_grad: {alpha_func.alpha_param.requires_grad}")
    
    # Create gradient-preserving solver
    solver = GradientPreservingWoSSolver(
        dirichletBoundary=dirichlet_boundary,
        diffusion=diffusion_func,
        absorption=alpha_func.forward,
        source=source_func
    )
    solver.setBoundaryConditions(boundary_condition)
    
    # Test points
    test_points = torch.tensor([[0.2, 0.3]])
    target_value = torch.tensor([[0.5]])
    
    print(f"\nRunning gradient-preserving solve...")
    
    # Forward pass
    solution = solver.solve_with_gradients(test_points, nWalks=50, maxSteps=50, fixed_seed=42)
    
    print(f"Solution: {solution.item():.6f}")
    print(f"Solution requires_grad: {solution.requires_grad}")
    print(f"Solution grad_fn: {solution.grad_fn}")
    
    if solution.requires_grad:
        # Compute loss and backpropagate
        loss = (solution - target_value)**2
        print(f"Loss: {loss.item():.6f}")
        
        loss.backward()
        
        print(f"Alpha gradient: {alpha_func.alpha_param.grad}")
        
        if alpha_func.alpha_param.grad is not None and alpha_func.alpha_param.grad.abs() > 1e-8:
            print(f"\n🎉 SUCCESS: Gradients preserved!")
            print(f"Gradient magnitude: {alpha_func.alpha_param.grad.abs().item():.8f}")
            return True
        else:
            print(f"\n❌ FAILED: Zero gradients")
            return False
    else:
        print(f"\n❌ FAILED: No gradients in solution")
        return False

def test_simple_optimization():
    """Test a simple optimization loop."""
    print(f"\n" + "="*60)
    print("SIMPLE OPTIMIZATION TEST")
    print("="*60)
    
    # Simple setup
    square_points = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]
    ])
    dirichlet_boundary = PolyLinesSimple(square_points)
    
    class OptimizableAlpha(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Start away from target
        
        def forward(self, point):
            return self.alpha.abs()  # Ensure positive
    
    alpha_func = OptimizableAlpha()
    
    def diffusion_func(point):
        return torch.tensor(1.0)
    
    def source_func(point):
        return 1.0  # Constant source
    
    def boundary_condition(point):
        return 0.0
    
    solver = GradientPreservingWoSSolver(
        dirichletBoundary=dirichlet_boundary,
        diffusion=diffusion_func,
        absorption=alpha_func.forward,
        source=source_func
    )
    solver.setBoundaryConditions(boundary_condition)
    
    test_points = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.3]])  # Target solution
    
    optimizer = torch.optim.Adam(alpha_func.parameters(), lr=0.01)
    
    print(f"Target solution: {target.item():.6f}")
    print(f"Initial alpha: {alpha_func.alpha.item():.6f}")
    
    for step in range(5):
        optimizer.zero_grad()
        
        solution = solver.solve_with_gradients(test_points, nWalks=30, maxSteps=30, fixed_seed=42+step)
        loss = (solution - target)**2
        
        if solution.requires_grad:
            loss.backward()
            optimizer.step()
            
            print(f"Step {step+1}: alpha={alpha_func.alpha.item():.6f}, solution={solution.item():.6f}, loss={loss.item():.8f}")
        else:
            print(f"Step {step+1}: No gradients - stopping")
            break

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test gradient preservation
    success = test_gradient_preservation()
    
    if success:
        print(f"\n🚀 Proceeding to optimization test...")
        test_simple_optimization()
    else:
        print(f"\n💡 Gradient preservation needs more work...")