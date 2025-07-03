import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from geometry.Polylines import PolyLinesSimple

def benchmark_jit_performance():
    """Benchmark the JIT-compiled geometry functions."""
    print("="*50)
    print("JIT COMPILATION PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Create a more complex polyline for benchmarking
    n_points = 100
    theta = torch.linspace(0, 2*torch.pi, n_points)
    radius = 1.0 + 0.3 * torch.sin(5 * theta)  # Star-like shape
    points = torch.stack([
        radius * torch.cos(theta),
        radius * torch.sin(theta)
    ], dim=1)
    
    polyline = PolyLinesSimple(points)
    
    # Create many test points for benchmarking
    n_tests = 1000
    test_points = torch.rand(n_tests, 2) * 2.0 - 1.0  # Random points in [-1, 1]
    directions = torch.rand(n_tests, 2) * 2.0 - 1.0   # Random directions
    
    print(f"Test configuration:")
    print(f"  Polyline vertices: {n_points}")
    print(f"  Test points: {n_tests}")
    print(f"  Total operations: {n_tests} per function")
    
    # Benchmark distance calculations
    print(f"\nBenchmarking distance calculations...")
    start_time = time.time()
    
    for i in range(n_tests):
        _ = polyline.distance(test_points[i])
    
    distance_time = time.time() - start_time
    distance_ops_per_sec = n_tests / distance_time
    
    print(f"  Distance calculations: {distance_time:.3f}s total")
    print(f"  Distance ops/sec: {distance_ops_per_sec:.0f}")
    
    # Benchmark ray intersections
    print(f"\nBenchmarking ray intersections...")
    start_time = time.time()
    
    for i in range(n_tests):
        _, _, _ = polyline.intersectPolylines(test_points[i], directions[i], 2.0)
    
    intersection_time = time.time() - start_time
    intersection_ops_per_sec = n_tests / intersection_time
    
    print(f"  Ray intersections: {intersection_time:.3f}s total")
    print(f"  Intersection ops/sec: {intersection_ops_per_sec:.0f}")
    
    # Warm-up complete - JIT should be fully compiled now
    print(f"\nPerformance Summary:")
    print(f"  Distance ops/sec: {distance_ops_per_sec:.0f}")
    print(f"  Intersection ops/sec: {intersection_ops_per_sec:.0f}")
    
    # Performance targets
    if distance_ops_per_sec > 10000:
        print("🚀 Excellent distance calculation performance!")
    elif distance_ops_per_sec > 5000:
        print("✅ Good distance calculation performance!")
    else:
        print("⚠️  Distance calculation could be faster")
    
    if intersection_ops_per_sec > 2000:
        print("🚀 Excellent intersection performance!")
    elif intersection_ops_per_sec > 1000:
        print("✅ Good intersection performance!")
    else:
        print("⚠️  Intersection calculation could be faster")
    
    return distance_ops_per_sec, intersection_ops_per_sec

def test_jit_correctness():
    """Test that JIT compilation maintains correctness."""
    print("\n" + "="*50)
    print("JIT CORRECTNESS VERIFICATION")
    print("="*50)
    
    # Create test polyline
    points = torch.tensor([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]
    ])
    polyline = PolyLinesSimple(points)
    
    # Test distance calculation
    test_point = torch.tensor([0.5, 0.5])
    distance = polyline.distance(test_point)
    expected_distance = 0.5
    assert torch.isclose(distance, torch.tensor(expected_distance), atol=1e-6), \
        f"Distance test failed: expected {expected_distance}, got {distance.item()}"
    print(f"✅ Distance calculation: {distance.item():.6f} (expected {expected_distance})")
    
    # Test ray intersection
    direction = torch.tensor([1.0, 0.0])
    intersection_point, normal, found = polyline.intersectPolylines(test_point, direction, 2.0)
    expected_intersection = torch.tensor([1.0, 0.5])
    
    assert found, "Ray intersection should be found"
    assert torch.allclose(intersection_point, expected_intersection, atol=1e-5), \
        f"Intersection point test failed: expected {expected_intersection}, got {intersection_point}"
    print(f"✅ Ray intersection: {intersection_point} (found={found})")
    
    # Test silhouette detection
    triangle_points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    triangle = PolyLinesSimple(triangle_points)
    test_point_sil = torch.tensor([1.5, 0.6])
    silhouette = triangle.isSilhouette(test_point_sil)
    expected_silhouette = torch.tensor([True])
    
    assert torch.equal(silhouette, expected_silhouette), \
        f"Silhouette test failed: expected {expected_silhouette}, got {silhouette}"
    print(f"✅ Silhouette detection: {silhouette}")
    
    print("\n✅ All JIT correctness tests passed!")

if __name__ == "__main__":
    # Test correctness first
    test_jit_correctness()
    
    # Then benchmark performance
    distance_perf, intersection_perf = benchmark_jit_performance()
    
    print(f"\n" + "="*50)
    print("JIT BENCHMARK COMPLETE")
    print("="*50)
    print(f"🔥 JIT compilation successfully implemented!")
    print(f"📊 Distance: {distance_perf:.0f} ops/sec")
    print(f"📊 Intersections: {intersection_perf:.0f} ops/sec")