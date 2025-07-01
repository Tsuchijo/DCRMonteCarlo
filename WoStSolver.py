import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Polylines import PolyLines, PolyLinesSimple

class WostSolver_2D:
    """
    A class to solve the forward problem for a 2D surface topology using the Walk on Spheres method.
    This class is designed to work with a 2D surface topology defined by a function and a boundary size.
    This generally solves for mixed boundary conditions for any elliptic pde defined by a laplace equation with a source, diffusion, and absorption term.
    """

    def __init__(self, dirichletBoundary: PolyLines, neumannBoundary: PolyLines = None, solvePoints: torch.Tensor = None,):
        """
        Initialize the solver with the surface topology and boundary size.
        
        Args:
            dirichletBoundary (PolyLines): The polyline representing the Dirichlet boundary.
            neumannBoundary (PolyLines, optional): The polyline representing the Neumann boundary.
            boundarySize (tuple[float, float]): The size of the boundary in the x and y dimensions.
            solvePoints (torch.Tensor, optional): Points where the solution is to be evaluated.
        """
        self.dirichletBoundary = dirichletBoundary
        self.neumannBoundary = neumannBoundary

        self.solve_points = solvePoints

        self.boundaryDirichlet = lambda point: 0.0  # Default Dirichlet boundary condition (can be set later)
        self.source = None
    
    def setBoundaryConditions(self, boundaryDirichlet: callable):
        """
        Set the boundary conditions for the solver.
        
        Args:
            boundaryDirichlet (callable): A function that takes a point and returns the value of the Dirichlet boundary condition at that point.
        """
        self.boundaryDirichlet = boundaryDirichlet

    def setSolvePoints(self, solve_points: torch.Tensor):
        """
        Set the points where the solution is to be evaluated.
        
        Args:
            solve_points (torch.Tensor): A tensor of points where the solution is to be evaluated.
        """
        self.solve_points = solve_points

    def setSourceTerm(self, source: callable):
        """
        Set the source term for the PDE.
        
        Args:
            source (callable): A function that takes a point and returns the value of the source term at that point.
        """
        self.source = source

    def greensFunction(self, x: torch.tensor, y: torch.Tensor, r: float) -> torch.Tensor:
        """
        Compute the Green's function for the Laplace equation in 2D on a circular domain.
        
        Args:
            x (torch.Tensor): The x-coordinates of the points.
            y (torch.Tensor): The y-coordinates of the points.
            r (float): The radius of the circular domain.
        
        Returns:
            torch.Tensor: The Green's function evaluated at the points (x, y).
        """
        r_xy = torch.norm(x - y, dim=1)
        if torch.any(r_xy == 0):
            raise ValueError("The points x and y must not be the same.")
        return torch.log(r / r_xy) / (2 * np.pi)  # Green's function for Laplace equation in 2D
    
    def greensFunctionIntegral(r: float) -> float:
        """
        Compute the integral of the Green's function over a circular domain.
        
        Args:
            r (float): The radius of the circular domain.
        
        Returns:
            float: The value of the integral.
        """
        return r**2 / 4  # Integral of the Green's function over a circular domain in 2D
    
    def sampleGreens(self, x: torch.tensor, r: float) -> float:
        """
        Randomly sample the source term weighted by the Green's function.
        This method chooses a random point on the circle of radius r centered on x weighted by the Green's function.
        returns a randomly sampled radius from the Green's function distribution.
        """
        return torch.sqrt(torch.rand(1)) * r  # Sample a radius from the Green's function distribution

    def _solveDirichlet(self, nWalks: int = 1000, maxSteps: int = 1000) -> torch.Tensor:
        """
        Simplified solve method for only Dirichlet boundary conditions (NeumannBoundayr=None).
        This is the basic implementation of the Walk on Spheres method.
        """

        eps = 1e-4
        rmin = 1e-3  # Minimum step size for the random walk
        results = torch.zeros((len(self.solve_points), 1))  # Initialize results tensor
        for i, point in enumerate(tqdm(self.solve_points, desc="Solving WoS", unit="pt")):
            # Initialize the random walk
            for _ in range(nWalks):
                current_point = point.clone()
                step_count = 0
                dDirichlet = 1.0  # Distance to the Dirichlet boundary

                while (step_count < maxSteps) & (dDirichlet > eps):
                    dDirichlet = self.dirichletBoundary.distance(current_point)
                    r = max(rmin, dDirichlet)  # Step size is the distance to the Dirichlet boundary
                    theta = torch.rand(1) * 2 * np.pi  # Random direction
                    direction = torch.tensor([torch.cos(theta), torch.sin(theta)])

                    if self.source is not None:
                        # If a source term is defined, sample the source term at the current point
                        r_sampled = self.sampleGreens(current_point, r)
                        sample_point  = current_point + r_sampled * direction                        
                        # accumulate the source term value at the sampled point with the Green's function
                        results[i] += self.source(sample_point) * self.greensFunction(current_point, sample_point, r_sampled)

                    current_point = current_point + r * direction  # Move in the random direction

                    step_count += 1  # Increment step count
                # After the random walk, accumulate the solution value at the point
                results[i] += self.boundaryDirichlet(current_point)  # Accumulate the solution value at the point

        return results / nWalks  # Return the average solution value at each point



    def solve(self, nWalks = 1000, maxSteps = 1000) -> torch.Tensor:
        """
        Solve the forward problem for the given surface topology and boundary size.
        
        Returns:
            torch.Tensor: A tensor of shape (N, 2) where N is the number of points in the solve_points tensor.
        """

        if self.neumannBoundary is None:
            # If no Neumann boundary is defined, use the Dirichlet solver
            return self._solveDirichlet(nWalks=nWalks, maxSteps=maxSteps)

        eps = 1e-4 # stopping tolerance
        rmin = 1e-3 # Minimum step size for the random walk
        results = torch.zeros((len(self.solve_points), 1)) # Initialize results tensor to store the accumulated solution values
        for i, point in enumerate(tqdm(self.solve_points, desc="Solving WoS", unit="pt")):
            # Initialize the random walk
            for _ in range(nWalks):
                current_point = point.clone()
                step_count = 0
                dDirichlet = 1.0 # Distance to the Dirichlet boundary
                onBoundary = False # Flag to check if the point is on the boundary
                normal = torch.tensor([1.0, 0.0]) # Normal vector at the intersection point
                while (step_count < maxSteps) & (dDirichlet > eps):
                    dDirichlet = self.dirichletBoundary.distance(current_point)
                    dNeumann = self.neumannBoundary.silhouetteDistance(current_point)
                    r = max(rmin, min(dDirichlet, dNeumann)) # Step size is the minimum distance to the boundaries
                    theta = torch.rand(1) * 2 * np.pi # Random direction
                    if onBoundary:
                        theta = theta/2 + torch.atan2(normal[1], normal[0]) # Reflect the direction if on the boundary
                    direction = torch.tensor([torch.cos(theta), torch.sin(theta)]) # Direction vector
                    next_point, normal, onBoundary = self.neumannBoundary.intersectPolylines(current_point, direction, r) # Get the next point and normal vector

                    if self.source is not None: # sample the source term
                        # If a source term is defined, sample the source term at the current point
                        r_sampled = self.sampleGreens(current_point, r)
                        sample_point  = current_point + r_sampled * direction                        
                        # accumulate the source term value at the sampled point with the Green's function
                        results[i] += self.source(sample_point) * self.greensFunction(current_point, sample_point, r_sampled)

                    current_point = next_point
                    step_count += 1 # Increment step count
                
                results[i] += self.boundaryDirichlet(current_point) # Accumulate the solution value at the point

        return results / nWalks # Return the average solution value at each point
    

