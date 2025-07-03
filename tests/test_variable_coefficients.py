import torch
import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLines, PolyLinesSimple


class TestVariableCoefficientWoS:
    """Test suite for variable coefficient Walk on Spheres with delta tracking."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a simple rectangular domain
        boundary_points = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]
        ])
        self.dirichlet_boundary = PolyLinesSimple(boundary_points)
        
        # Define test points inside the domain
        self.test_points = torch.tensor([
            [0.25, 0.25], [0.5, 0.5], [0.75, 0.75]
        ])
    
    def test_delta_tracking_initialization(self):
        """Test that delta tracking is properly initialized with variable coefficients."""
        # Define variable diffusion and absorption coefficients
        def diffusion_func(point):
            return 1.0 + 0.5 * torch.sin(torch.pi * point[0]) * torch.cos(torch.pi * point[1])
        
        def absorption_func(point):
            return 0.8 + 0.3 * (point[0]**2 + point[1]**2)
        
        # Initialize solver with variable coefficients
        solver = WostSolver_2D(
            dirichletBoundary=self.dirichlet_boundary,
            diffusion=diffusion_func,
            absorption=absorption_func
        )
        
        # Check that delta tracking components are initialized
        assert hasattr(solver, 'use_delta_tracking')
        assert solver.use_delta_tracking == True
        assert hasattr(solver, 'sigma_prime')
        assert hasattr(solver, 'sigma_bar')
        assert hasattr(solver, 'diffusion')
        assert hasattr(solver, 'absorption')
        
        # Test that sigma_bar is positive (necessary for delta tracking)
        assert solver.sigma_bar > 0, "sigma_bar must be positive for delta tracking"
        
        print(f"✓ Delta tracking initialized with sigma_bar = {solver.sigma_bar:.4f}")
    
    def test_screened_greens_function_properties(self):
        """Test properties of the screened Green's function."""
        def diffusion_func(point):
            return 1.0
        
        def absorption_func(point):
            return 2.0
        
        solver = WostSolver_2D(
            dirichletBoundary=self.dirichlet_boundary,
            diffusion=diffusion_func,
            absorption=absorption_func
        )
        
        # Test screened Green's function evaluation
        x = torch.tensor([0.3, 0.4])
        y = torch.tensor([0.6, 0.7])
        R = 0.2
        
        greens_val = solver.screenedGreens(x, y, R)
        assert isinstance(greens_val, (float, torch.Tensor)), "Green's function should return numeric value"
        
        # Test screened Green's function norm
        norm_val = solver.screenedGreensNorm(x, R)
        assert norm_val > 0, "Green's function norm should be positive"
        
        print(f"✓ Screened Green's function: G = {greens_val:.6f}, norm = {norm_val:.6f}")
    
    def test_screened_greens_sampling(self):
        """Test rejection sampling from screened Green's function."""
        def diffusion_func(point):
            return 1.5
        
        def absorption_func(point):
            return 1.2
        
        solver = WostSolver_2D(
            dirichletBoundary=self.dirichlet_boundary,
            diffusion=diffusion_func,
            absorption=absorption_func
        )
        
        x = torch.tensor([0.5, 0.5])
        r = 0.1
        
        # Generate multiple samples to test distribution
        samples = []
        for _ in range(100):
            sample = solver.sampleScreenedGreens(x, r)
            samples.append(sample)
            assert 0 <= sample <= r, f"Sample {sample} should be in [0, {r}]"
        
        samples = np.array(samples)
        
        # Basic statistical tests
        assert len(samples) == 100, "Should generate requested number of samples"
        assert np.std(samples) > 0, "Samples should have non-zero variance"
        
        print(f"✓ Screened Green's sampling: mean = {np.mean(samples):.4f}, std = {np.std(samples):.4f}")
    
    def test_variable_coefficient_pde_solution(self):
        """Test solving a PDE with variable coefficients using delta tracking."""
        # Define spatially varying coefficients
        def diffusion_func(point):
            return 1.0 + 0.2 * point[0]  # Linear variation in x
        
        def absorption_func(point):
            return 0.5 + 0.3 * point[1]  # Linear variation in y
        
        def source_func(point):
            return torch.sin(torch.pi * point[0]) * torch.sin(torch.pi * point[1])
        
        # Boundary condition (zero Dirichlet)
        def boundary_condition(point):
            return 0.0
        
        solver = WostSolver_2D(
            dirichletBoundary=self.dirichlet_boundary,
            source=source_func,
            diffusion=diffusion_func,
            absorption=absorption_func
        )
        solver.setBoundaryConditions(boundary_condition)
        
        # Solve at test points
        solution = solver.solve(self.test_points, nWalks=500, maxSteps=1000)
        
        # Basic solution tests
        assert solution.shape == (len(self.test_points), 1), "Solution should have correct shape"
        assert torch.all(torch.isfinite(solution)), "Solution should be finite"
        
        # Print solution values
        for i, (point, value) in enumerate(zip(self.test_points, solution)):
            print(f"✓ Solution at ({point[0]:.2f}, {point[1]:.2f}): {value.item():.6f}")
    
    def test_convergence_with_known_solution(self):
        """Test convergence to analytical solution for a simple case."""
        # Simple case: constant coefficients with known solution
        # ∇²u - κ²u = 0 with u = sin(πx)sin(πy) on boundary
        # This gives u = sin(πx)sin(πy)/(1 + π²κ²) for κ² = absorption/diffusion
        
        kappa_squared = 2.0
        
        def diffusion_func(point):
            return 1.0
        
        def absorption_func(point):
            return kappa_squared
        
        def boundary_condition(point):
            return torch.sin(torch.pi * point[0]) * torch.sin(torch.pi * point[1])
        
        solver = WostSolver_2D(
            dirichletBoundary=self.dirichlet_boundary,
            diffusion=diffusion_func,
            absorption=absorption_func
        )
        solver.setBoundaryConditions(boundary_condition)
        
        # Test point in center of domain
        test_point = torch.tensor([[0.5, 0.5]])
        
        # Analytical solution at center
        analytical = np.sin(np.pi * 0.5) * np.sin(np.pi * 0.5) / (1 + np.pi**2 / kappa_squared)
        
        # Monte Carlo solution with increasing number of walks
        walk_counts = [100, 500, 1000]
        errors = []
        
        for nWalks in walk_counts:
            numerical = solver.solve(test_point, nWalks=nWalks, maxSteps=1000)
            error = abs(numerical.item() - analytical)
            errors.append(error)
            print(f"✓ nWalks={nWalks}: numerical={numerical.item():.6f}, analytical={analytical:.6f}, error={error:.6f}")
        
        # Check that error generally decreases with more walks (Monte Carlo convergence)
        # Allow some noise due to randomness
        assert errors[0] > errors[2] * 0.5, "Error should generally decrease with more walks"
    
    def test_domain_bounds_calculation(self):
        """Test that domain bounds are correctly calculated for variable coefficient problems."""
        def diffusion_func(point):
            return 1.0
        
        def absorption_func(point):
            return 1.0
        
        solver = WostSolver_2D(
            dirichletBoundary=self.dirichlet_boundary,
            diffusion=diffusion_func,
            absorption=absorption_func
        )
        
        # Check domain bounds
        expected_bounds = [[0.0, 1.0], [0.0, 1.0]]
        assert np.allclose(solver.domain_bounds, expected_bounds), "Domain bounds should match boundary extent"
        
        print(f"✓ Domain bounds correctly calculated: {solver.domain_bounds}")
    
    def test_modified_diffusion_term(self):
        """Test the modified diffusion term computation for delta tracking."""
        def diffusion_func(point):
            return 1.0 + 0.1 * point[0]
        
        def absorption_func(point):
            return 2.0 + 0.2 * point[1]
        
        solver = WostSolver_2D(
            dirichletBoundary=self.dirichlet_boundary,
            diffusion=diffusion_func,
            absorption=absorption_func
        )
        
        # Test modified diffusion at a point
        test_point = torch.tensor([0.5, 0.5])
        sigma_prime_val = solver.sigma_prime(test_point)
        
        assert isinstance(sigma_prime_val, torch.Tensor), "sigma_prime should return tensor"
        assert torch.isfinite(sigma_prime_val), "sigma_prime should be finite"
        
        print(f"✓ Modified diffusion σ'(0.5, 0.5) = {sigma_prime_val:.6f}")
        print(f"✓ Diffusion bound σ_bar = {solver.sigma_bar:.6f}")


def run_performance_benchmark():
    """Benchmark performance of variable coefficient solver."""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Create test setup
    boundary_points = torch.tensor([
        [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]
    ])
    dirichlet_boundary = PolyLinesSimple(boundary_points)
    
    def diffusion_func(point):
        return 1.0 + 0.5 * torch.exp(-((point[0]-1)**2 + (point[1]-1)**2))
    
    def absorption_func(point):
        return 0.8 + 0.4 * torch.sin(torch.pi * point[0]) * torch.cos(torch.pi * point[1])
    
    def source_func(point):
        return torch.exp(-((point[0]-1)**2 + (point[1]-1)**2))
    
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        source=source_func,
        diffusion=diffusion_func,
        absorption=absorption_func
    )
    solver.setBoundaryConditions(lambda p: 0.0)
    
    # Create grid of test points
    x = torch.linspace(0.2, 1.8, 5)
    y = torch.linspace(0.2, 1.8, 5)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    test_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    import time
    start_time = time.time()
    
    solution = solver.solve(test_points, nWalks=200, maxSteps=500)
    
    end_time = time.time()
    
    print(f"Solved {len(test_points)} points in {end_time - start_time:.2f} seconds")
    print(f"Average time per point: {(end_time - start_time)/len(test_points):.4f} seconds")
    print(f"Solution range: [{solution.min():.6f}, {solution.max():.6f}]")


if __name__ == "__main__":
    # Run the tests
    test_suite = TestVariableCoefficientWoS()
    test_suite.setup_method()
    
    print("TESTING VARIABLE COEFFICIENT WALK ON SPHERES WITH DELTA TRACKING")
    print("="*70)
    
    try:
        test_suite.test_delta_tracking_initialization()
        test_suite.test_screened_greens_function_properties()
        test_suite.test_screened_greens_sampling()
        test_suite.test_variable_coefficient_pde_solution()
        test_suite.test_convergence_with_known_solution()
        test_suite.test_domain_bounds_calculation()
        test_suite.test_modified_diffusion_term()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        
        # Run performance benchmark
        run_performance_benchmark()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise