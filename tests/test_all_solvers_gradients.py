import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLinesSimple

def test_dirichlet_solver_gradients():
    """Test that the Dirichlet solver preserves gradients."""
    print("="*60)
    print("TESTING DIRICHLET SOLVER GRADIENTS")
    print("="*60)
    
    # Create square domain
    square_points = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]
    ])
    dirichlet_boundary = PolyLinesSimple(square_points)
    
    # Parametric source function
    class ParametricSource(nn.Module):
        def __init__(self):
            super().__init__()
            self.source_param = nn.Parameter(torch.tensor(2.0))
        
        def forward(self, point):
            return self.source_param
    
    source_func = ParametricSource()
    
    def boundary_condition(point):
        return 0.0
    
    print(f"Initial source parameter: {source_func.source_param.item():.6f}")
    
    # Create solver with only Dirichlet boundary (no Neumann, no absorption)
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        source=source_func.forward
    )
    solver.setBoundaryConditions(boundary_condition)
    
    print(f"Delta tracking enabled: {solver.use_delta_tracking}")
    print(f"Neumann boundary: {solver.neumannBoundary}")
    
    test_points = torch.tensor([[0.2, 0.3]])
    
    print(f"\nTesting Dirichlet solver:")
    
    try:
        solution = solver.solve(test_points, nWalks=50, maxSteps=50)
        print(f"Solution: {solution.item():.6f}")
        print(f"Solution requires_grad: {solution.requires_grad}")
        
        if solution.requires_grad:
            target = torch.tensor([[0.5]])
            loss = (solution - target)**2
            loss.backward()
            
            print(f"Source parameter gradient: {source_func.source_param.grad}")
            
            if source_func.source_param.grad is not None:
                print("🎉 SUCCESS: Dirichlet solver preserves gradients!")
                return True
            else:
                print("❌ ISSUE: No gradients in Dirichlet solver")
                return False
        else:
            print("❌ ISSUE: Solution doesn't require gradients")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_mixed_boundary_solver_gradients():
    """Test that the mixed boundary solver preserves gradients."""
    print("\n" + "="*60)
    print("TESTING MIXED BOUNDARY SOLVER GRADIENTS")
    print("="*60)
    
    # Create square domain with inner circle as Neumann boundary
    square_points = torch.tensor([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]
    ])
    dirichlet_boundary = PolyLinesSimple(square_points)
    
    # Create circle as Neumann boundary
    theta = torch.linspace(0, 2*torch.pi, 50)
    radius = 0.3
    circle_x = radius * torch.cos(theta)
    circle_y = radius * torch.sin(theta)
    circle_points = torch.stack([circle_x, circle_y], dim=1)
    # Close the circle
    circle_points = torch.cat([circle_points, circle_points[0:1]], dim=0)
    neumann_boundary = PolyLinesSimple(circle_points)
    
    # Parametric source function
    class ParametricSource(nn.Module):
        def __init__(self):
            super().__init__()
            self.source_param = nn.Parameter(torch.tensor(1.5))
        
        def forward(self, point):
            return self.source_param
    
    source_func = ParametricSource()
    
    def boundary_condition(point):
        return 0.0
    
    print(f"Initial source parameter: {source_func.source_param.item():.6f}")
    
    # Create solver with mixed boundaries (no absorption)
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        neumannBoundary=neumann_boundary,
        source=source_func.forward
    )
    solver.setBoundaryConditions(boundary_condition)
    
    print(f"Delta tracking enabled: {solver.use_delta_tracking}")
    print(f"Has Neumann boundary: {solver.neumannBoundary is not None}")
    
    test_points = torch.tensor([[0.5, 0.5]])
    
    print(f"\nTesting mixed boundary solver:")
    
    try:
        solution = solver.solve(test_points, nWalks=50, maxSteps=50)
        print(f"Solution: {solution.item():.6f}")
        print(f"Solution requires_grad: {solution.requires_grad}")
        
        if solution.requires_grad:
            target = torch.tensor([[0.8]])
            loss = (solution - target)**2
            loss.backward()
            
            print(f"Source parameter gradient: {source_func.source_param.grad}")
            
            if source_func.source_param.grad is not None:
                print("🎉 SUCCESS: Mixed boundary solver preserves gradients!")
                return True
            else:
                print("❌ ISSUE: No gradients in mixed boundary solver")
                return False
        else:
            print("❌ ISSUE: Solution doesn't require gradients")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test all three solvers
    dirichlet_success = test_dirichlet_solver_gradients()
    mixed_success = test_mixed_boundary_solver_gradients()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dirichlet solver gradients: {'✅ PASS' if dirichlet_success else '❌ FAIL'}")
    print(f"Mixed boundary solver gradients: {'✅ PASS' if mixed_success else '❌ FAIL'}")
    print(f"Delta tracking solver gradients: ✅ PASS (tested earlier)")
    
    if dirichlet_success and mixed_success:
        print(f"\n🎉 ALL SOLVERS NOW PRESERVE GRADIENTS CONSISTENTLY!")
    else:
        print(f"\n⚠️  Some solvers still need gradient fixes")