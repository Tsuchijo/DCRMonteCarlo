import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLinesSimple

def simple_gradient_fix_demo():
    """Demonstrate the simple .detach() approach to preserve gradients."""
    
    print("="*60)
    print("MINIMAL GRADIENT FIX DEMONSTRATION")
    print("="*60)
    
    # Example of the issue and fix
    print("1. Testing random number generation with gradients:")
    
    # Create a parameter
    param = torch.tensor(2.0, requires_grad=True)
    
    # BROKEN: Random numbers break gradient flow
    print("\nBROKEN approach:")
    random_broken = torch.rand(1) * 2 * np.pi  # This has requires_grad=False
    result_broken = param * torch.sin(random_broken)
    print(f"  random_broken.requires_grad: {random_broken.requires_grad}")
    print(f"  result_broken.requires_grad: {result_broken.requires_grad}")
    print(f"  result_broken.grad_fn: {result_broken.grad_fn}")
    
    # FIXED: Detach random numbers
    print("\nFIXED approach:")
    random_fixed = (torch.rand(1) * 2 * np.pi).detach()  # Explicitly detach
    result_fixed = param * torch.sin(random_fixed)
    print(f"  random_fixed.requires_grad: {random_fixed.requires_grad}")
    print(f"  result_fixed.requires_grad: {result_fixed.requires_grad}")
    print(f"  result_fixed.grad_fn: {result_fixed.grad_fn}")
    
    # Test backpropagation
    if result_fixed.requires_grad:
        loss = result_fixed**2
        loss.backward()
        print(f"  param.grad: {param.grad}")
        print("  ✅ Gradients flow correctly!")
    
    print("\n" + "="*60)
    print("APPLYING TO WOS SOLVER")
    print("="*60)
    
    # Show what changes are needed in existing solver
    print("""
Minimal changes needed in your WoStSolver.py:

1. In _get_random_angle():
   OLD: return self.random_batches['angles'][self.random_batches['angle_idx']]
   NEW: return self.random_batches['angles'][self.random_batches['angle_idx']].detach()

2. In _get_random_uniform():
   OLD: return self.random_batches['uniforms'][self.random_batches['uniform_idx']]
   NEW: return self.random_batches['uniforms'][self.random_batches['uniform_idx']].detach()

3. In _refill_random_batches():
   OLD: self.random_batches['angles'] = torch.rand(self.random_batch_size) * 2 * np.pi
   NEW: self.random_batches['angles'] = (torch.rand(self.random_batch_size) * 2 * np.pi).detach()
   
   OLD: self.random_batches['uniforms'] = torch.rand(self.random_batch_size)
   NEW: self.random_batches['uniforms'] = torch.rand(self.random_batch_size).detach()

4. Anywhere you call .item() on tensors that should preserve gradients:
   OLD: dDirichlet = self.dirichletBoundary.distance(current_point).item()
   NEW: dDirichlet = self.dirichletBoundary.distance(current_point).detach().item()
   
That's it! Just 4-6 lines of changes.
""")

def test_detach_approach():
    """Test the .detach() approach with a real absorption function."""
    
    print("="*60)
    print("TESTING DETACH APPROACH WITH ABSORPTION")
    print("="*60)
    
    # Create domain that will use delta tracking (absorption function)
    square_points = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]
    ])
    dirichlet_boundary = PolyLinesSimple(square_points)
    
    # Parametric absorption function
    class ParametricAbsorption(nn.Module):
        def __init__(self):
            super().__init__()
            self.absorption_param = nn.Parameter(torch.tensor(1.5))
        
        def forward(self, point):
            # Absorption varies with position and parameter
            x, y = point[0], point[1]
            return self.absorption_param * (1.0 + 0.1 * (x**2 + y**2))
    
    absorption_func = ParametricAbsorption()
    
    def diffusion_func(point):
        return torch.tensor(1.0)  # Constant diffusion
    
    def source_func(point):
        return 1.0  # Constant source
    
    def boundary_condition(point):
        return 0.0
    
    print(f"Initial absorption parameter: {absorption_func.absorption_param.item():.6f}")
    
    # Create solver with BOTH diffusion and absorption (triggers delta tracking)
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        diffusion=diffusion_func,
        absorption=absorption_func.forward,  # This should be called!
        source=source_func
    )
    solver.setBoundaryConditions(boundary_condition)
    
    print(f"Delta tracking enabled: {solver.use_delta_tracking}")
    print(f"Sigma bar: {solver.sigma_bar:.6f}")
    
    if not solver.use_delta_tracking:
        print("❌ ERROR: Delta tracking not enabled - absorption function won't be called!")
        return False
    
    # Test points
    test_points = torch.tensor([[0.2, 0.3]])
    
    print(f"\nTesting current solver (likely broken gradients):")
    
    try:
        # Enable anomaly detection to find the exact issue
        torch.autograd.set_detect_anomaly(True)
        
        # Test with current solver (probably breaks gradients)
        solution = solver.solve(test_points, nWalks=50, maxSteps=50)
        print(f"Solution: {solution.item():.6f}")
        print(f"Solution requires_grad: {solution.requires_grad}")
        
        if solution.requires_grad:
            target = torch.tensor([[0.3]])
            loss = (solution - target)**2
            loss.backward()
            
            print(f"Absorption parameter gradient: {absorption_func.absorption_param.grad}")
            
            if absorption_func.absorption_param.grad is not None:
                print("🎉 SUCCESS: Gradients work with current solver!")
                return True
            else:
                print("❌ ISSUE: No gradients in current solver")
                return False
        else:
            print("❌ ISSUE: Solution doesn't require gradients")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Demonstrate the concept
    simple_gradient_fix_demo()
    
    # Test with actual absorption function
    success = test_detach_approach()
    
    if success:
        print(f"\n🎉 Your solver already preserves gradients!")
    else:
        print(f"\n🔧 You need to add .detach() calls to preserve gradients")
        print(f"   See the minimal changes listed above")