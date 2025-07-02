import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import i0, k0
from Polylines import PolyLines, PolyLinesSimple
from utils import torchGradient, torchLaplacian, gridSampleMinMax

class WostSolver_2D:
    """
    A class to solve the forward problem for a 2D surface topology using the Walk on Spheres method.
    This class is designed to work with a 2D surface topology defined by a function and a boundary size.
    This generally solves for mixed boundary conditions for any elliptic pde defined by a laplace equation with a source, diffusion, and absorption term.
    """

    def __init__(self, dirichletBoundary: PolyLines, neumannBoundary: PolyLines = None, source: callable = None, diffusion: callable = None, absorption: callable = None):
        """
        Initialize the solver with the surface topology and boundary size.
        
        Args:
            dirichletBoundary (PolyLines): The polyline representing the Dirichlet boundary.
            neumannBoundary (PolyLines, optional): The polyline representing the Neumann boundary.
            boundarySize (tuple[float, float]): The size of the boundary in the x and y dimensions.
        """
        self.dirichletBoundary = dirichletBoundary
        self.neumannBoundary = neumannBoundary

        # get the min and max bounds of the domain by looking for the min and max of each component of both boundaries
        min_x = min(dirichletBoundary.points[:, 0].min(), neumannBoundary.points[:, 0].min()) if neumannBoundary else dirichletBoundary.points[:, 0].min()
        max_x = max(dirichletBoundary.points[:, 0].max(), neumannBoundary.points[:, 0].max()) if neumannBoundary else dirichletBoundary.points[:, 0].max()
        min_y = min(dirichletBoundary.points[:, 1].min(), neumannBoundary.points[:, 1].min()) if neumannBoundary else dirichletBoundary.points[:,   1].min()
        max_y = max(dirichletBoundary.points[:, 1].max(), neumannBoundary.points[:, 1].max()) if neumannBoundary else dirichletBoundary.points[:, 1 ].max()

        self.domain_bounds = [[min_x, max_x], [min_y, max_y]]  # Store the domain bounds for later use

        self.boundaryDirichlet = lambda point: 0.0  # Default Dirichlet boundary condition (can be set later)
        self.source = source

        # Diffusion or absorption term is provided create the modified terms needed for delta tracking
        if diffusion is not None or absorption is not None:
            if diffusion is None:
                diffusion = lambda point: 0.0
            if absorption is None:
                absorption = lambda point: 1.0

            self.absorption = absorption
            self.diffusion = diffusion
            # Build the modified diffusion term for delta tracking
            self.sigma_prime, self.sigma_bar = self.buildModifiedDiffusion()
            self.use_delta_tracking = True
    
    def buildModifiedDiffusion(self):
        """
        Build the modified diffusion term for the delta tracking method.
        Args:
            diffusion (callable): A function that takes a point and returns the diffusion value at that point.
        returns:
            callable: A function that computes the modified diffusion term at a given point.
            float: The estimated difference between the maximum and minimum of the modified diffusion term over the domain.
        """
        # Wrap the original functions to ensure they return tensors for autograd compatibility
        def diffusion_wrapped(point):
            result = self.diffusion(point)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=torch.float32)
            return result
        
        def absorption_wrapped(point):
            result = self.absorption(point)
            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=torch.float32)
            return result
        
        def sigma_prime(point):
            """
            Modified diffusion term for delta tracking.
            defined from Sawhney et al. 2023
            $$\sigma = \frac{\sigma(x)}{\alpha(x)} + \frac{1}{2}(\frac{\laplaca\alpha(x)}{\alpha(x)} - \frac{|\nabla ln(\alpha(x))|^2}{2})$$
            """
            # Ensure point has gradients enabled for autograd
            point = point.clone().detach().requires_grad_(True)
            
            try:
                # Compute Laplacian of absorption function
                alpha_laplacian = torchLaplacian(absorption_wrapped, point)
                
                # Create a log-absorption function for gradient computation
                def log_absorption(p):
                    return torch.log(absorption_wrapped(p))
                
                alpha_log_grad = torchGradient(log_absorption, point)
                alpha_log_grad_norm = alpha_log_grad.norm() ** 2

                return (diffusion_wrapped(point) / absorption_wrapped(point)) + \
                       0.5 * (alpha_laplacian / absorption_wrapped(point) - \
                               alpha_log_grad_norm / 2.0)
            except Exception as e:
                # If gradient computation fails, fall back to simple ratio
                return diffusion_wrapped(point) / absorption_wrapped(point)
        
        # find the sigma bar term, which should be the estimated difference between the min and max of the modified diffusion term on the domain
        try:
            min_sigma, max_sigma, _, _ = gridSampleMinMax(sigma_prime, self.domain_bounds, grid_resolution=50)
            sigma_bar = max_sigma - min_sigma
            
            # Ensure sigma_bar is positive and reasonable
            if sigma_bar <= 0:
                sigma_bar = 1.0  # fallback value
                
        except Exception as e:
            # If grid sampling fails, use a simple heuristic
            print(f"Warning: Could not compute sigma_bar via grid sampling: {e}")
            sigma_bar = 1.0  # fallback value

        return sigma_prime, sigma_bar
    
    def screenedGreensNorm(self, point: torch.Tensor, r: float) -> float:
        """
        Compute the normalized screened Green's function for the given point and radius.
        This is used to sample the source term weighted by the Green's function.
        
        Args:
            point (torch.Tensor): The point at which to evaluate the Green's function.
            r (float): The radius for the Green's function.
        
        Returns:
            float: The normalized value of the Green's function at the given point and radius.
        """
        I_0 = i0(r * np.sqrt(self.sigma_bar))  # Modified Bessel function of the first kind
        return 1 / self.sigma_bar * (1 - 1/I_0)
    
    def screenedGreens(self, x: torch.Tensor, y: torch.Tensor, R: float) -> float:
        """
        Compute the screened Green's function for the given point and radius.
        This is used to sample the source term weighted by the Green's function.
        
        Args:
            x (torch.Tensor): The x-coordinate of the point.
            y (torch.Tensor): The y-coordinate of the point.
            R (float): The radius for the Green's function.
        
        Returns:
            float: The value of the Green's function at the given point and radius.
        """
        
        r = (x - y).norm()  # Compute the distance between the points
        I_0_R = i0(R * np.sqrt(self.sigma_bar))  # Modified Bessel function of the first kind
        I_0_r = i0(r * np.sqrt(self.sigma_bar))  # Modified Bessel function of the first kind
        K_0_R = k0(R * np.sqrt(self.sigma_bar))  # Modified Bessel function of the second kind
        K_0_r = k0(r * np.sqrt(self.sigma_bar))  # Modified Bessel function of the second kind

        return 1/(2 * np.pi) * (K_0_r - (K_0_R / I_0_R) * I_0_r)

    def sampleScreenedGreens(self, x: torch.Tensor, r: float) -> float:
        """
        Randomly sample the source term weighted by the screened Green's function.
        This method chooses a random point on the circle of radius r centered on x weighted by the Green's function.
        
        Args:
            x (torch.Tensor): The center point for sampling.
            r (float): The radius for sampling.
        
        Returns:
            float: A randomly sampled radius from the Green's function distribution.
        """
        # Initialize cache if it doesn't exist
        if not hasattr(self, 'screened_greens_cache'):
            self.screened_greens_cache = []
            self.cache_index = 0
            self.cache_size = 10000
        # Refill cache if empty
        if len(self.screened_greens_cache) == 0 or self.cache_index >= len(self.screened_greens_cache):
            self._refill_screened_greens_cache()
            self.cache_index = 0
        
        # Get next cached sample and scale by current radius
        normalized_sample = self.screened_greens_cache[self.cache_index]
        self.cache_index += 1
        
        # Scale the normalized sample (which is in [0,1]) to current radius
        return normalized_sample * r

    def _refill_screened_greens_cache(self):
        """
        Refill the cache with samples from the screened Green's function distribution using rejection sampling.
        Samples are normalized to [0,1] and will be scaled by the actual radius when used.
        Uses the screened Green's function and its norm for rejection sampling.
        """
        samples = []
        # Use the screened Green's function norm as the envelope for rejection sampling
        max_density = self.screenedGreensNorm(torch.zeros(2), 1.0)  # Maximum density at r=1
        
        while len(samples) < self.cache_size:
            # Sample radius uniformly in (0, 1]
            r_candidate = np.random.uniform(1e-6, 1.0)
            
            # Create dummy points for Green's function evaluation
            x_dummy = torch.zeros(2)
            y_dummy = torch.tensor([r_candidate, 0.0])
            
            # Evaluate the screened Green's function density
            density_val = abs(self.screenedGreens(x_dummy, y_dummy, 1.0))
            
            # Rejection step: accept sample if uniform random < density/envelope
            if np.random.uniform(0, max_density) < density_val:
                samples.append(r_candidate)
        
        self.screened_greens_cache = samples

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

    
    def sampleGreens(self, x: torch.tensor, r: float) -> float:
        """
        Randomly sample the source term weighted by the Green's function.
        This method chooses a random point on the circle of radius r centered on x weighted by the Green's function.
        returns a randomly sampled radius from the Green's function distribution.
        """
        # Initialize cache if it doesn't exist
        if not hasattr(self, 'greens_cache'):
            self.greens_cache = []
            self.cache_index = 0
            self.cache_size = 10000  # Number of samples to pre-generate
        
        # Refill cache if empty
        if len(self.greens_cache) == 0 or self.cache_index >= len(self.greens_cache):
            self._refill_greens_cache()
            self.cache_index = 0
        
        # Get next cached sample and scale by current radius
        normalized_sample = self.greens_cache[self.cache_index]
        self.cache_index += 1
        
        # Scale the normalized sample (which is in [0,1]) to current radius
        return normalized_sample * r

    def _refill_greens_cache(self):
        """
        Refill the cache with samples from the Green's function distribution using rejection sampling.
        Samples are normalized to [0,1] and will be scaled by the actual radius when used.
        """
        samples = []
        max_density = 1.0  # log(1/epsilon) where epsilon → 0, but we'll use log(1/small_val)
        small_val = 1e-6  # To avoid log(0)
        max_log_val = -np.log(small_val)  # This is our envelope height
        
        while len(samples) < self.cache_size:
            # Sample radius uniformly in (0, 1]
            r_candidate = np.random.uniform(small_val, 1.0)
            
            # Evaluate density: f(r) ∝ log(1/r) = -log(r)
            density_val = -np.log(r_candidate)
            
            # Rejection step
            if np.random.uniform(0, max_log_val) < density_val:
                samples.append(r_candidate)
        
        self.greens_cache = samples

    def _solveDirichlet(self, solvePoints, nWalks: int = 1000, maxSteps: int = 1000) -> torch.Tensor:
        """
        Simplified solve method for only Dirichlet boundary conditions (NeumannBoundayr=None).
        This is the basic implementation of the Walk on Spheres method.
        """

        eps = 1e-4
        rmin = 1e-3  # Minimum step size for the random walk
        results = torch.zeros((len(solvePoints), 1))  # Initialize results tensor
        for i, point in enumerate(tqdm(solvePoints, desc="Solving WoS", unit="pt")):
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
                        results[i] += self.source(sample_point) * r**2 / 4 # scale by normalization factor of the Green's function


                    current_point = current_point + r * direction  # Move in the random direction

                    step_count += 1  # Increment step count
                # After the random walk, accumulate the solution value at the point
                results[i] += self.boundaryDirichlet(current_point)  # Accumulate the solution value at the point

        return results / nWalks  # Return the average solution value at each point

    def _solve_delta_tacking(self, solvePoints: torch.tensor, nWalks = 1000, maxSteps = 1000) -> torch.Tensor:
        """
        More advanced solver that uses the delta tracking method to solve the PDE with spacially varying diffusion and absorption coefficients.
        """
        eps = 1e-4 # stopping tolerance
        rmin = 1e-6 # Minimum step size for the random walk
        results = torch.zeros((len(solvePoints), 1)) # Initialize results tensor to store the accumulated solution values
        for i, point in enumerate(tqdm(solvePoints, desc="Solving WoS", unit="pt")):
            # Initialize the random walk
            for _ in range(nWalks):
                current_point = point.clone()
                step_count = 0
                dDirichlet = 1.0 # Distance to the Dirichlet boundary
                onBoundary = False # Flag to check if the point is on the boundary
                normal = torch.tensor([1.0, 0.0]) # Normal vector at the intersection point
                attenuation_coef = 1.0 # accumulate attenuation factors onto this every time we branch
                while (step_count < maxSteps) & (dDirichlet > eps):
                    dDirichlet = self.dirichletBoundary.distance(current_point)
                    dNeumann = self.neumannBoundary.silhouetteDistance(current_point)
                    r = max(rmin, min(dDirichlet, dNeumann)) # Step size is the minimum distance to the boundaries
                    theta = torch.rand(1) * 2 * np.pi # Random direction
                    if onBoundary:
                        theta = theta/2 + torch.atan2(normal[1], normal[0]) # Reflect the direction if on the boundary
                    direction = torch.tensor([torch.cos(theta), torch.sin(theta)]) # Direction vector
                    next_point, normal, onBoundary = self.neumannBoundary.intersectPolylines(current_point, direction, r) # Get the next point and normal vector

                    r_sampled = self.sampleScreenedGreens(current_point, r) # Sample the Green's function at the current point
                    sample_point  = current_point + r_sampled * direction

                    # if we try to sample from outside the domain, just set the sample_point to be the next point (hacky fix)
                    if (sample_point - current_point).norm() > (next_point - current_point).norm():
                        sample_point = next_point

                    if self.source is not None: # sample the source term
                        # If a source term is defined, sample the source term at the current point
                        # accumulate the source term value at the sampled point with the Green's function
                        results[i] += (self.source(sample_point) * self.screenedGreensNorm(current_point, r) \
                        / torch.sqrt(self.absorption(sample_point) * self.absorption(current_point))) * attenuation_coef
                
                    # sample randomly from 0-1
                    mu = torch.rand(1)
                    greens_norm = self.screenedGreensNorm(current_point, r)
                    if mu > self.sigma_bar * greens_norm:
                        # sample from the edge of the sphere
                        attenuation_coef *= torch.sqrt(self.absorption(next_point) / self.absorption(current_point))
                        current_point = next_point
                    
                    else:
                        sigma_prime_val = self.sigma_prime(sample_point)
                        attenuation_coef *= torch.sqrt(self.absorption(sample_point) / self.absorption(current_point)) \
                            * (1 - sigma_prime_val / self.sigma_bar)
                        current_point = sample_point


                    step_count += 1 # Increment step count
                
                results[i] += self.boundaryDirichlet(current_point) * attenuation_coef# Accumulate the solution value at the point

        return results / nWalks # Return the average solution value at each point
    


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

        if self.neumannBoundary is None:
            # If no Neumann boundary is defined, use the Dirichlet solver
            return self._solveDirichlet(solvePoints, nWalks=nWalks, maxSteps=maxSteps)

        if self.use_delta_tracking:
            return self._solve_delta_tacking(solvePoints, nWalks=nWalks, maxSteps=maxSteps)

        eps = 1e-4 # stopping tolerance
        rmin = 1e-6 # Minimum step size for the random walk
        results = torch.zeros((len(solvePoints), 1)) # Initialize results tensor to store the accumulated solution values
        for i, point in enumerate(tqdm(solvePoints, desc="Solving WoS", unit="pt")):
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
                        if not ((sample_point - current_point).norm() > (next_point - current_point).norm()):
                            # If the sampled point is further than the next point we are outside the domain and should not sample it
                            # accumulate the source term value at the sampled point with the Green's function
                            results[i] += self.source(sample_point) * r**2 / 4 # scale by normalization factor of the Green's function

                    current_point = next_point
                    step_count += 1 # Increment step count
                
                results[i] += self.boundaryDirichlet(current_point) # Accumulate the solution value at the point

        return results / nWalks # Return the average solution value at each point
    

