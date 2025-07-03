import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from solvers.WoStSolver import WostSolver_2D
from geometry.Polylines import PolyLinesSimple

def create_test_case():
    """Create a simple test case for benchmarking."""
    # Simple square domain
    boundary_points = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]
    ])
    dirichlet_boundary = PolyLinesSimple(boundary_points)
    
    # Simple boundary condition
    def boundary_condition(point):
        return float(point[0] + point[1])
    
    # Simple source term
    def source_term(point):
        return 1.0
    
    # Test points
    test_points = torch.tensor([
        [0.25, 0.25], [0.5, 0.5], [0.75, 0.75]
    ])
    
    return dirichlet_boundary, boundary_condition, source_term, test_points

def benchmark_solver():
    """Benchmark the optimized solver performance."""
    print("="*50)
    print("PERFORMANCE BENCHMARK - TENSOR PREALLOCATION")
    print("="*50)
    
    # Create test case
    dirichlet_boundary, boundary_condition, source_term, test_points = create_test_case()
    
    # Initialize solver
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        source=source_term
    )
    solver.setBoundaryConditions(boundary_condition)
    
    # Test parameters
    nWalks = 200
    maxSteps = 100
    
    print(f"Test configuration:")
    print(f"  Points: {len(test_points)}")
    print(f"  Walks per point: {nWalks}")
    print(f"  Max steps: {maxSteps}")
    print(f"  Total walk iterations: {len(test_points) * nWalks}")
    
    # Warm up the solver (initialize caches, etc.)
    print("\nWarming up solver...")
    _ = solver.solve(test_points[:1], nWalks=10, maxSteps=10)
    
    # Benchmark the solve
    print("\nRunning benchmark...")
    start_time = time.time()
    
    solution = solver.solve(test_points, nWalks=nWalks, maxSteps=maxSteps)
    
    end_time = time.time()
    solve_time = end_time - start_time
    
    # Calculate performance metrics
    total_walks = len(test_points) * nWalks
    walks_per_second = total_walks / solve_time
    time_per_walk = solve_time / total_walks * 1000  # milliseconds
    
    print(f"\nPerformance Results:")
    print(f"  Total solve time: {solve_time:.3f} seconds")
    print(f"  Walks per second: {walks_per_second:.0f}")
    print(f"  Time per walk: {time_per_walk:.3f} ms")
    print(f"  Solution range: [{solution.min():.4f}, {solution.max():.4f}]")
    
    # Memory usage estimation
    tensor_memory = solver.max_walks * 2 * 4 * 6  # 6 tensors, 2D, float32
    batch_memory = solver.random_batch_size * 4 * 2  # 2 batches, float32
    total_memory_kb = (tensor_memory + batch_memory) / 1024
    
    print(f"\nMemory Usage (Preallocated):")
    print(f"  Work tensors: {tensor_memory / 1024:.1f} KB")
    print(f"  Random batches: {batch_memory / 1024:.1f} KB")
    print(f"  Total preallocation: {total_memory_kb:.1f} KB")
    
    return solve_time, walks_per_second

def test_correctness():
    """Test that optimizations maintain correctness."""
    print("\n" + "="*50)
    print("CORRECTNESS TEST")
    print("="*50)
    
    # Create test case
    dirichlet_boundary, boundary_condition, source_term, test_points = create_test_case()
    
    # Create solver
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        source=source_term
    )
    solver.setBoundaryConditions(boundary_condition)
    
    # Run multiple times to check consistency
    solutions = []
    for i in range(3):
        torch.manual_seed(42 + i)  # Different seeds for each run
        solution = solver.solve(test_points, nWalks=50, maxSteps=50)
        solutions.append(solution)
        print(f"Run {i+1}: {solution.flatten().tolist()}")
    
    # Check that solutions are finite and reasonable
    for i, sol in enumerate(solutions):
        assert torch.all(torch.isfinite(sol)), f"Solution {i+1} contains non-finite values"
        assert torch.all(sol >= 0), f"Solution {i+1} contains negative values (unexpected for this problem)"
        assert torch.all(sol <= 10), f"Solution {i+1} contains unreasonably large values"
    
    print("✅ All correctness tests passed!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run correctness test
    test_correctness()
    
    # Run performance benchmark
    solve_time, walks_per_second = benchmark_solver()
    
    print(f"\n" + "="*50)
    print("BENCHMARK COMPLETE")
    print("="*50)
    print(f"Performance: {walks_per_second:.0f} walks/sec")
    
    # Performance targets (these are rough estimates)
    if walks_per_second > 1000:
        print("🚀 Excellent performance!")
    elif walks_per_second > 500:
        print("✅ Good performance!")
    elif walks_per_second > 100:
        print("⚠️  Moderate performance")
    else:
        print("❌ Performance needs improvement")