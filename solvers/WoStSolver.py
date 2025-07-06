import torch
import numpy as np
from tqdm import tqdm
from scipy.special import i0, k0
from geometry.Polylines import PolyLines
from utils import torchGradient, torchLaplacian, gridSampleMinMax
from .utils import (
    GreensDistribution2D,
    ScreenedGreensDistribution2D,
    screenedGreensNorm2D,
    greensFunctionNorm2D
)


class WostSolver_2D:
    """
    A class to solve the forward problem for a 2D surface topology using the Walk on Spheres method.
    This class is designed to work with a 2D surface topology defined by a function and a boundary size.
    This generally solves for mixed boundary conditions for any elliptic pde defined by a laplace equation with a source, diffusion, and absorption term.
    """

    def __init__(self, dirichletBoundary: PolyLines, dirichletBoundaryFunction: callable = None, neumannBoundary: PolyLines = None, source: callable = None, sigma: callable = None, alpha: callable = None):
        """
        Initialize the solver with the surface topology and boundary size.
        
        Args:
            dirichletBoundary (PolyLines): The polyline representing the Dirichlet boundary.
            neumannBoundary (PolyLines, optional): The polyline representing the Neumann boundary.
            boundarySize (tuple[float, float]): The size of the boundary in the x and y dimensions.
            Source (callable): the source function over the domain
            sigma (callable): the absorption function over the domain
            alpha (callable): the diffusion fucntion over the domain
        """
        self.dirichletBoundary = dirichletBoundary
        self.neumannBoundary = neumannBoundary

        # get the min and max bounds of the domain by looking for the min and max of each component of both boundaries
        min_x = min(dirichletBoundary.points[:, 0].min(), neumannBoundary.points[:, 0].min()) if neumannBoundary else dirichletBoundary.points[:, 0].min()
        max_x = max(dirichletBoundary.points[:, 0].max(), neumannBoundary.points[:, 0].max()) if neumannBoundary else dirichletBoundary.points[:, 0].max()
        min_y = min(dirichletBoundary.points[:, 1].min(), neumannBoundary.points[:, 1].min()) if neumannBoundary else dirichletBoundary.points[:,   1].min()
        max_y = max(dirichletBoundary.points[:, 1].max(), neumannBoundary.points[:, 1].max()) if neumannBoundary else dirichletBoundary.points[:, 1 ].max()

        self.domain_bounds = [[min_x, max_x], [min_y, max_y]]  # Store the domain bounds for later use

        if dirichletBoundaryFunction is None:
            self.boundaryDirichlet = lambda point: 0.0  # Default Dirichlet boundary condition (can be set later)
        else:
            self.boundaryDirichlet = dirichletBoundaryFunction
            
        self.source = source
        self.use_delta_tracking = False

        # Diffusion or absorption term is provided create the modified terms needed for delta tracking
        if sigma is not None or alpha is not None:
            if sigma is None:
                sigma = lambda point: 0.0
            if alpha is None:
                alpha = lambda point: 1.0

            self.alpha = alpha
            self.sigma = sigma
            # Build the modified absorption term for delta tracking
            self.sigma_prime, self.sigma_bar = self.buildModifiedSigma()
            self.use_delta_tracking = True
    
    def buildModifiedSigma(self):
        """
        Build the modified Sigma term for the delta tracking method.
        returns:
            callable: A function that computes the modified absorption term at a given point.
            float: The estimated difference between the maximum and minimum of the modified absorption term over the domain.
        """
        # Wrap the original functions to ensure they return tensors for autograd compatibility
        def sigma_wrapped(point):
            result = self.sigma(point)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=torch.float32, requires_grad=True)
            return result
        
        def alpha_wrapped(point):
            result = self.alpha(point)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=torch.float32, requires_grad=True)
            # Ensure alpha is always positive to avoid numerical issues
            epsilon = 1e-8
            return torch.clamp(result, min=epsilon)
        
        def sigma_prime(point):
            """
            Modified absorption term for delta tracking.
            defined from Sawhney et al. 2023
            $$\sigma = \frac{\sigma(x)}{\alpha(x)} + \frac{1}{2}(\frac{\laplaca\alpha(x)}{\alpha(x)} - \frac{|\nabla ln(\alpha(x))|^2}{2})$$
            """
            # Ensure point has gradients enabled while preserving computation graph
            if not point.requires_grad:
                point = point.clone().requires_grad_(True)
            else:
                # If already has gradients, just clone to avoid modifying the original
                point = point.clone()
            
            # Simple fallback computation
            simple_ratio = sigma_wrapped(point) / alpha_wrapped(point)
            try:
                # Compute Laplacian of alpha function
                alpha_laplacian = torchLaplacian(alpha_wrapped, point)
                
                # Create a log-alpha function for gradient computation
                def log_alpha(p):
                    alpha_val = alpha_wrapped(p)
                    # Add small epsilon to prevent log(0) issues
                    epsilon = 1e-8
                    return torch.log(alpha_val + epsilon)
                
                alpha_log_grad = torchGradient(log_alpha, point)
                alpha_log_grad_norm = (alpha_log_grad ** 2).sum()
                
                # Compute the full modified sigma
                alpha_val = alpha_wrapped(point)
                correction_term = 0.5 * (alpha_laplacian / alpha_val - alpha_log_grad_norm / 2.0)
                
                return simple_ratio + correction_term
                
            except Exception as e:
                # If gradient computation fails, fall back to simple ratio
                # This is actually fine for many cases where alpha is constant or nearly 
                print(f"Failed with exception {e}")
                return simple_ratio
        
        # find the sigma bar term, which should be the estimated difference between the min and max of the modified absorption term on the domain
        min_sigma, max_sigma, _, _ = gridSampleMinMax(sigma_prime, self.domain_bounds, grid_resolution=50)
        sigma_bar = max_sigma
        
        # Ensure sigma_bar is positive and reasonable
        if (sigma_bar <= 0) | (sigma_bar > 1e6):
            print("Sigma_bar is too small, falling back on value")
            sigma_bar = 1.0  # fallback value
                
        return sigma_prime, sigma_bar
    

    def setBoundaryConditions(self, boundaryDirichlet: callable):
        """
        Set the boundary conditions for the solver.
        
        Args:
            boundaryDirichlet (callable): A function that takes a point and returns the value of the Dirichlet boundary condition at that point.
        """
        self.boundaryDirichlet = boundaryDirichlet

    def setSourceTerm(self, source: callable):
        """
        Set the source term for the PDE.
        
        Args:
            source (callable): A function that takes a point and returns the value of the source term at that point.
        """
        self.source = source

    


    def _solveUnified(self, solvePoints, nWalks: int = 1000, maxSteps: int = 1000) -> torch.Tensor:
        """
        Unified solver that handles all cases: Dirichlet-only, mixed boundaries, and delta tracking.
        Uses conditional logic to skip unnecessary calculations for simpler cases.
        """
        eps = 1e-4
        rmin = 1e-6 # rmin must be less than eps, or else risk suddenly jumping outside of the domain resulting in odd behavior 
        sampler = GreensDistribution2D(10000)

        # Determine solver mode for progress bar description
        if self.use_delta_tracking:
            desc = "Solving WoS Delta (Unified)"
            sampler = ScreenedGreensDistribution2D(self.sigma_bar)
        elif self.neumannBoundary is not None:
            desc = "Solving WoS Mixed (Unified)"
        else:
            desc = "Solving WoS Dirichlet (Unified)"
        
        results_list = []
        
        for point in tqdm(solvePoints, desc=desc, unit="pt"):
            point_total = torch.tensor(0.0, requires_grad=True)
            
            for i in range(nWalks):
                current_point = point.clone()
                step_count = 0
                dDirichlet = 1.0
                
                # Delta tracking variables (only used if delta tracking is enabled)
                onBoundary = False
                normal = torch.tensor([0, 1])
                attenuation_coef = torch.tensor(1.0) if self.use_delta_tracking else None

                while (step_count < maxSteps) & (dDirichlet > eps):
                    # Calculate distances to boundaries
                    dDirichlet = self.dirichletBoundary.distance(current_point)
                    
                    if self.neumannBoundary is not None:
                        dNeumann = self.neumannBoundary.silhouetteDistance(current_point)
                        r = max(rmin, min(dDirichlet, dNeumann))
                    else:
                        r = max(rmin, dDirichlet)
                    
                    # Generate random direction
                    theta = torch.rand(1) * 2 * np.pi
                    if onBoundary and self.neumannBoundary is not None:
                        theta = theta/2 + torch.atan2(normal[1], normal[0])
                    
                    cos_theta = torch.cos(theta)
                    sin_theta = torch.sin(theta)
                    direction = torch.tensor([cos_theta, sin_theta])
                    
                    # Determine next point based on boundary conditions
                    if self.neumannBoundary is not None:
                        next_point, normal, onBoundary = self.neumannBoundary.intersectPolylines(current_point, direction, r)
                    else:
                        next_point = current_point + r * direction
                        onBoundary = False
                    
                    # Handle source term sampling
                    if self.source is not None:
                        # Delta tracking: sample from screened Green's function
                        r_sampled = sampler.sample(current_point, r)
                        sample_point = current_point + r_sampled * direction
                        
                        # Clamp sample point to domain
                        if (sample_point - current_point).norm() > (next_point - current_point).norm():
                            sample_point = next_point
                            source_contribution = 0 # if outside domain dont count source contribution
                        
                        elif self.use_delta_tracking:
                            source_contribution = (self.source(sample_point) * screenedGreensNorm2D(r, self.sigma_bar) 
                                                / torch.sqrt(self.alpha(sample_point) * self.alpha(current_point))) * attenuation_coef
                        else:
                            source_contribution = self.source(sample_point) * greensFunctionNorm2D(r)

                        point_total = point_total + source_contribution

                    
                    # Move to next point based on delta tracking or standard method
                    if self.use_delta_tracking:
                        mu = torch.rand(1).item()
                        greens_norm = screenedGreensNorm2D(r, self.sigma_bar)
                        
                        if mu > self.sigma_bar * greens_norm:
                            # Sample from edge of sphere
                            attenuation_coef = attenuation_coef * torch.sqrt(self.alpha(next_point) / self.alpha(current_point))
                            current_point = next_point.clone()
                        else:
                            # Sample from interior using delta tracking
                            sigma_prime_val = self.sigma_prime(sample_point)
                            sigma_scaling = max((1 - sigma_prime_val / self.sigma_bar), 0.0)
                            attenuation_coef = (attenuation_coef * torch.sqrt(self.alpha(sample_point) / self.alpha(current_point)) * sigma_scaling)
                            current_point = sample_point.clone()
                    else:
                        # Standard method: move to next point
                        current_point = next_point.clone()
                    
                    step_count += 1
                
                # Add boundary contribution
                boundary_contribution = self.boundaryDirichlet(current_point)
                if self.use_delta_tracking:
                    boundary_contribution = boundary_contribution * attenuation_coef
                point_total = (point_total + boundary_contribution)
            
            results_list.append(point_total / nWalks)
        
        return torch.stack(results_list).unsqueeze(1)

    #@torch.no_grad()
    def solve(self, solvePoints: torch.tensor, nWalks = 1000, maxSteps = 1000) -> torch.Tensor:
        """
        Solve the forward problem for the given surface topology and boundary size.

        This method uses the Walk on Spheres method to solve the PDE defined by the surface topology and boundary conditions.
        Args:
            solvePoints (torch.Tensor): Points where the solution is to be evaluated. Shape should be
            (N, 2) where N is the number of points.
            nWalks (int): Number of random walks to perform for each point.
            maxSteps (int): Maximum number of steps for each random walk.
        
        Returns:
            torch.Tensor: A tensor of shape (N, 2) where N is the number of points in the solve_points tensor.
        """

        # Use unified solver for all cases
        return self._solveUnified(solvePoints, nWalks=nWalks, maxSteps=maxSteps)