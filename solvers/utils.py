from scipy.special import i0, k0
import torch
import numpy as np

def screenedGreens2D(x: torch.Tensor, y: torch.Tensor, R: float, sigmaBar: float) -> float:
    """
    Compute the screened Green's function for the given point and radius.
    This is used to sample the source term weighted by the Green's function.
    
    Args:
        x (torch.Tensor): The x-coordinate of the point.
        y (torch.Tensor): The y-coordinate of the point.
        R (float): The radius for the Green's function.
        sigmaBar (float): the sigma screening parameter for the greens function
    
    Returns:
        float: The value of the Green's function at the given point and radius.
    """
    
    r = (x - y).norm()  # Compute the distance between the points
    I_0_R = i0(R * np.sqrt(sigmaBar))  # Modified Bessel function of the first kind
    I_0_r = i0(r * np.sqrt(sigmaBar))  # Modified Bessel function of the first kind
    K_0_R = k0(R * np.sqrt(sigmaBar))  # Modified Bessel function of the second kind
    K_0_r = k0(r * np.sqrt(sigmaBar))  # Modified Bessel function of the second kind

    return 1/(2 * np.pi) * (K_0_r - (K_0_R / I_0_R) * I_0_r)


def screenedGreensNorm2D(R: float, sigmaBar: float) -> float:
    """
    Compute the normalization factor of the screened Green's function for a given screening parameter and radius.
    This is used to sample the source term weighted by the Green's function.
    
    Args:
        x (torch.Tensor): The point at which to evaluate the Green's function.
        R (float): The radius for the Green's function.
        sigmaBar (float): the sigma screening parameter for the greens function

    Returns:
        float: The normalized value of the Green's function at the given point and radius.
    """

    I_0 = i0(R * np.sqrt(sigmaBar))  # Modified Bessel function of the first kind
    return 1 / sigmaBar * (1 - 1/I_0)

def greensFunction2D(x: torch.Tensor, y: torch.Tensor, R: float) -> float:
    """
    Compute the Green's function in 2D for a circular domain for 2 given points and a radius 
    This implements the standard 2D Green's function: G(x,y) = -1/(2π) * log(|x-y|)
    """
    r = (x - y).norm()
    if r < 1e-10:
        return 0.0
    return -1.0 / (2.0 * np.pi) * torch.log(r)

def greensFunctionNorm2D(R: float) -> float:
    """
    Compute the normalization factor of the greens function in 2D
    This is the integral of the Green's function over a disk of radius R
    """
    return R**2 / 4


class SamplingDistribution2D:
    """
    Base class for 2D sampling distributions used in Green's function sampling.
    This enables multiple importance sampling by allowing different sampling strategies.
    
    Note: This is the 2D version. A future SamplingDistribution3D class can be 
    implemented for 3D solvers.
    """
    
    def __init__(self, cache_size: int = 10000):
        self.cache_size = cache_size
        self.cache = []
        self.cache_index = 0
    
    def sample(self, center: torch.Tensor, radius: float) -> float:
        """
        Sample a radius from this distribution.
        
        Args:
            center: Center point for sampling
            radius: Maximum radius for sampling
            
        Returns:
            Sampled radius value
        """
        raise NotImplementedError
    
    def pdf(self, r: float, center: torch.Tensor, radius: float) -> float:
        """
        Probability density function for this distribution.
        
        Args:
            r: Radius value to evaluate
            center: Center point
            radius: Maximum radius
            
        Returns:
            PDF value at r
        """
        raise NotImplementedError
    
    def _refill_cache(self):
        """Refill the sampling cache."""
        raise NotImplementedError
    
    def _get_cached_sample(self, radius: float) -> float:
        """Get a cached sample scaled by radius."""
        if len(self.cache) == 0 or self.cache_index >= len(self.cache):
            self._refill_cache()
            self.cache_index = 0
        
        normalized_sample = self.cache[self.cache_index]
        self.cache_index += 1
        return normalized_sample * radius


class GreensDistribution2D(SamplingDistribution2D):
    """
    Sampling distribution for the standard 2D Green's function.
    Uses rejection sampling with density proportional to -log(r).
    
    For 3D solvers, implement GreensDistribution3D with appropriate 3D Green's function.
    """
    
    def sample(self, center: torch.Tensor, radius: float) -> float:
        """Sample from Green's function distribution."""
        return self._get_cached_sample(radius)
    
    def pdf(self, r: float, center: torch.Tensor, radius: float) -> float:
        """PDF proportional to -log(r/radius)."""
        if r <= 0 or r >= radius:
            return 0.0
        return -np.log(r / radius) / (radius**2 / 4)
    
    def _refill_cache(self):
        """Refill cache using rejection sampling."""
        samples = []
        small_val = 1e-6
        max_log_val = -np.log(small_val)
        
        while len(samples) < self.cache_size:
            r_candidate = np.random.uniform(small_val, 1.0)
            density_val = -np.log(r_candidate)
            
            if np.random.uniform(0, max_log_val) < density_val:
                samples.append(r_candidate)
        
        self.cache = samples


class ScreenedGreensDistribution2D(SamplingDistribution2D):
    """
    Sampling distribution for the 2D screened Green's function.
    Uses rejection sampling with the screened Green's function density.
    
    For 3D solvers, implement ScreenedGreensDistribution3D with 3D screened Green's function.
    """
    
    def __init__(self, sigma_bar: float, cache_size: int = 10000):
        super().__init__(cache_size)
        self.sigma_bar = sigma_bar
    
    def sample(self, center: torch.Tensor, radius: float) -> float:
        """Sample from screened Green's function distribution."""
        return self._get_cached_sample(radius)
    
    def pdf(self, r: float, center: torch.Tensor, radius: float) -> float:
        """PDF based on screened Green's function."""
        if r <= 0 or r >= radius:
            return 0.0
        
        x_dummy = torch.zeros(2)
        y_dummy = torch.tensor([r, 0.0])
        density = abs(screenedGreens2D(x_dummy, y_dummy, radius, self.sigma_bar))
        norm = screenedGreensNorm2D(radius, self.sigma_bar)
        return density / norm
    
    def _refill_cache(self):
        """Refill cache using rejection sampling."""
        samples = []
        max_density = screenedGreensNorm2D(1.0, self.sigma_bar)
        
        while len(samples) < self.cache_size:
            r_candidate = np.random.uniform(1e-6, 1.0)
            
            x_dummy = torch.zeros(2)
            y_dummy = torch.tensor([r_candidate, 0.0])
            density_val = abs(screenedGreens2D(x_dummy, y_dummy, 1.0, self.sigma_bar))
            
            if np.random.uniform(0, max_density) < density_val:
                samples.append(r_candidate)
        self.cache = samples


class UniformDistribution2D(SamplingDistribution2D):
    """
    Uniform sampling distribution for 2D domains for comparison and multiple importance sampling.
    
    For 3D solvers, implement UniformDistribution3D with appropriate 3D sampling.
    """
    
    def sample(self, center: torch.Tensor, radius: float) -> float:
        """Sample uniformly from [0, radius]."""
        return np.random.uniform(0, radius)
    
    def pdf(self, r: float, center: torch.Tensor, radius: float) -> float:
        """Uniform PDF."""
        if 0 <= r <= radius:
            return 1.0 / radius
        return 0.0
    
    def _refill_cache(self):
        """Uniform samples don't need caching."""
        pass


class MultipleImportanceSampler2D:
    """
    Multiple importance sampling coordinator for 2D Green's function sampling.
    Combines different 2D sampling strategies with optimal weights.
    
    For 3D solvers, implement MultipleImportanceSampler3D with 3D distributions.
    """
    
    def __init__(self, distributions: list, weights: list = None):
        """
        Initialize with list of sampling distributions and optional weights.
        
        Args:
            distributions: List of SamplingDistribution2D objects
            weights: Optional list of weights for each distribution
        """
        self.distributions = distributions
        self.weights = weights if weights else [1.0 / len(distributions)] * len(distributions)
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)  # Normalize
    
    def sample(self, center: torch.Tensor, radius: float) -> tuple:
        """
        Sample using multiple importance sampling.
        
        Returns:
            Tuple of (sampled_radius, distribution_index, mis_weight)
        """
        # Choose distribution based on weights
        dist_idx = np.random.choice(len(self.distributions), p=self.weights)
        
        # Sample from chosen distribution
        sampled_r = self.distributions[dist_idx].sample(center, radius)
        
        # Compute MIS weight
        mis_weight = self._compute_mis_weight(sampled_r, center, radius, dist_idx)
        
        return sampled_r, dist_idx, mis_weight
    
    def _compute_mis_weight(self, r: float, center: torch.Tensor, radius: float, sampled_idx: int) -> float:
        """
        Compute the multiple importance sampling weight using balance heuristic.
        
        Args:
            r: Sampled radius
            center: Center point
            radius: Maximum radius
            sampled_idx: Index of distribution that generated the sample
            
        Returns:
            MIS weight
        """
        # Get PDF values from all distributions
        pdf_values = []
        for dist in self.distributions:
            pdf_values.append(dist.pdf(r, center, radius))
        
        pdf_values = np.array(pdf_values)
        
        # Balance heuristic: w_i = (n_i * p_i) / sum(n_j * p_j)
        weighted_pdfs = self.weights * pdf_values
        denominator = np.sum(weighted_pdfs)
        
        if denominator == 0:
            return 0.0
        
        return weighted_pdfs[sampled_idx] / denominator


def sampleGreensFunction2D(center: torch.Tensor, radius: float, distribution: SamplingDistribution2D = None) -> float:
    """
    Sample a radius from the Green's function distribution.
    
    Args:
        center: Center point for sampling
        radius: Maximum radius for sampling
        distribution: Optional distribution to use (defaults to GreensDistribution)
        
    Returns:
        Sampled radius value
    """
    if distribution is None:
        distribution = GreensDistribution2D()
    
    return distribution.sample(center, radius)


def sampleScreenedGreensFunction2D(center: torch.Tensor, radius: float, sigma_bar: float, 
                                 distribution: ScreenedGreensDistribution2D = None) -> float:
    """
    Sample a radius from the 2D screened Green's function distribution.
    
    Args:
        center: Center point for sampling
        radius: Maximum radius for sampling
        sigma_bar: Screening parameter
        distribution: Optional distribution to use
        
    Returns:
        Sampled radius value
    """
    if distribution is None:
        distribution = ScreenedGreensDistribution2D(sigma_bar)
    
    return distribution.sample(center, radius)


# TODO: Future 3D implementations
# def sampleGreensFunction3D(center: torch.Tensor, radius: float, distribution: SamplingDistribution3D = None) -> float:
#     """Sample a radius from the 3D Green's function distribution."""
#     pass
#
# def sampleScreenedGreensFunction3D(center: torch.Tensor, radius: float, sigma_bar: float, 
#                                  distribution: ScreenedGreensDistribution3D = None) -> float:
#     """Sample a radius from the 3D screened Green's function distribution.""" 
#     pass
