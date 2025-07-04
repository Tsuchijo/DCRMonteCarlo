import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLinesSimple

def create_test_domain():
    """
    Create a test domain with dirichlet boundaries on 3 sides and a neumann boundary on the top
    """

    dirichlet_points = PolyLinesSimple(torch.tensor([
       [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],  [1.0, 1.0]
    ]))

    neumann_points = PolyLinesSimple(torch.tensor([
        [-1.0, 1.0], [1.0, 1.0]
    ]))

    return dirichlet_points, neumann_points


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
    
def source(p):
    ## source is just a small positive box on the surface
    x, y = p[0], p[1]
    if (y > 0.9) & (abs(x) < 0.1):
        return 1.0
    return 0.0

@torch.no_grad()
def test():

    alpha_func = ParametricAbsorption(
        base_alpha=1.0,
        amplitude=0.5, 
        center_x=0.0,
        center_y=0.0,
        width=0.3
    )

    dirichlet_points, neumann_points = create_test_domain()
    solver = WostSolver_2D(
        dirichletBoundary= dirichlet_points,
        neumannBoundary= neumann_points,
        source=source,
        diffusion = lambda x: 0.0,
        absorption=alpha_func
    )

    solver.boundaryDirichlet = lambda x: 0.0

    test_points = torch.linspace(-1.0, 1.0, 10).unsqueeze(1)
    test_points = torch.cat([test_points, torch.ones(10,1)], dim=1)

    solution = solver.solve(test_points, nWalks=20, maxSteps=1000)

    plt.plot(solution.detach())
    plt.show()

if __name__ == "__main__":
    test()

