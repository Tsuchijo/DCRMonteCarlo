import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers.WoStSolver import WostSolver_2D
from geometry.PolylinesSimple import PolyLinesSimple
from utils import torch_smooth_circle

def dcr_current_source(point):
    """
    DCR survey current source with positive and negative electrode pair.
    """
    x, y = point[0], point[1]
    
    # Current parameters
    current_amplitude = 1.0
    electrode_separation = 20.0
    sigma = 0.5
    
    # Positive electrode at (-10, 0), negative at (10, 0) 
    pos_dist2 = (x + 10.0)**2 + y**2
    neg_dist2 = (x - 10.0)**2 + y**2
    
    # Gaussian approximation for electrode current density
    norm = current_amplitude / (2 * torch.pi * sigma**2)
    
    # Positive source - negative sink
    positive_source = norm * torch.exp(-pos_dist2 / (2 * sigma**2))
    negative_sink = -norm * torch.exp(-neg_dist2 / (2 * sigma**2))
    
    return float(positive_source - negative_sink)

def conductivity_field(point):
    """
    Define a smooth, differentiable resistivity field with background and two conductive anomalies.
    Uses smooth exponential transitions instead of hard boundaries.
    """
    x, y = point[0], point[1]
    
    # Background conductivity (1/100 ohm-m)
    background_conductivity = 1e2
    anomaly_center1 = torch.tensor([-20, -30])
    anomaly_center2 = torch.tensor([25, -40])
    anomaly_value1 = (1e1 - background_conductivity )
    anomaly_value2 = (1e3 - background_conductivity)

    anomaly1 = anomaly_value1 * torch_smooth_circle(point, anomaly_center1, 10)
    anomaly2 = anomaly_value2 * torch_smooth_circle(point, anomaly_center2, 10)

    # Total conductivity: background + anomalies
    total_conductivity = background_conductivity + anomaly1 + anomaly2
    
    return total_conductivity


def create_surface_measurement_grid(x_range=(-50, 50), y_surface=0.0, spacing=5.0):
    """
    Create a grid of measurement electrodes on the surface topology.
    
    Parameters:
    - x_range: tuple (x_min, x_max) for measurement line extent
    - y_surface: y-coordinate of surface (0 = surface level)
    - spacing: distance between measurement electrodes
    
    Returns:
    - tensor: measurement electrode positions
    """
    x_positions = torch.arange(x_range[0], x_range[1] + spacing, spacing)
    y_positions = torch.full_like(x_positions, y_surface)
    
    measurement_points = torch.stack([x_positions, y_positions], dim=1)
    return measurement_points


def run_dcr_survey_simulation():
    """
    Run complete DCR survey simulation with conductive anomalies.
    """
    print("Setting up DCR survey simulation...")
    
    # Domain configuration
    domain_size = 200.0  # 200m x 200m survey area
    half_size = domain_size / 2.0

    # Create Dirichlet boundary (bottom and sides, excluding top surface)
    dirichlet_points = torch.tensor([
        [-half_size, -half_size],  # Bottom left
        [half_size, -half_size],   # Bottom right
        [half_size, half_size],    # Top right
        [-half_size, half_size],   # Top left
        [-half_size, -half_size]   # Close the boundary
    ])
    


    # Create Neumann boundary (top surface only)
    neumann_points = torch.tensor([
        [-half_size, half_size],   # Top left
        [half_size, half_size]     # Top right
    ])
    
    # Create boundary geometries
    dirichlet_boundary = PolyLinesSimple(dirichlet_points)
    neumann_boundary = PolyLinesSimple(neumann_points)
    
    # Surface measurement electrodes (every 10m along surface, smaller range)
    measurement_electrodes = create_surface_measurement_grid(
        x_range=(-40, 40), 
        y_surface=0.0, 
        spacing=10.0
    )
    
    print(f"Created {len(measurement_electrodes)} measurement electrodes")
    print(f"Electrode positions range: x=[{measurement_electrodes[:, 0].min():.1f}, {measurement_electrodes[:, 0].max():.1f}]")
    
    # Current injection configuration
    current_amplitude = 1.0  # 1 Ampere
    electrode_separation = 20.0  # 20m between current electrodes
    
    print(f"Current injection: {current_amplitude}A between electrodes at ±{electrode_separation/2:.1f}m")
    print(f"Conductive anomalies: Sphere 1 at (-20, -30)m, Sphere 2 at (25, -40)m")
    
    # Zero voltage boundary condition for Dirichlet boundaries (far field)
    def dirichlet_bc(point):
        return 0.0
    
    # Initialize solver
    print("Initializing WoS solver...")
    solver = WostSolver_2D(
        dirichletBoundary=dirichlet_boundary,
        dirichletBoundaryFunction=dirichlet_bc,
        neumannBoundary=neumann_boundary,  # Top surface with zero flux (insulating)
        source=dcr_current_source,
        alpha=conductivity_field,  # Conductivity field
        sigma=None  # No absorption for DC resistivity
    )
    
    # Run simulation
    print("Running DCR survey simulation...")
    n_walks = 100  # Reduced for testing
    max_steps = 500
    
    voltages = solver.solve(
        measurement_electrodes,
        nWalks=n_walks,
        maxSteps=max_steps,
        eps = 1.0
    )
    
    print(f"Simulation completed. Voltage range: [{voltages.min():.6f}, {voltages.max():.6f}] V")
    
    return measurement_electrodes, voltages

def plot_dcr_survey_results(measurement_positions, measured_voltages, save_plot=False):
    """
    Create visualization of DCR survey results.
    """
    try:
        # Create surface voltage profile
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Surface voltage measurements
        x_positions = measurement_positions[:, 0].detach().numpy()
        voltages = measured_voltages.detach().numpy()
        
        ax1.plot(x_positions, voltages, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Distance (m)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('DCR Survey: Surface Voltage Measurements')
        ax1.grid(True, alpha=0.3)
        
        # Mark current injection points
        ax1.axvline(x=-10, color='red', linestyle='--', alpha=0.7, label='Current injection (+)')
        ax1.axvline(x=10, color='darkred', linestyle='--', alpha=0.7, label='Current injection (-)')
        
        # Mark anomaly locations (projected to surface)
        ax1.axvline(x=-20, color='green', linestyle=':', alpha=0.7, label='Anomaly 1')
        ax1.axvline(x=25, color='darkgreen', linestyle=':', alpha=0.7, label='Anomaly 2')
        ax1.legend()
        
        # Plot 2: Subsurface resistivity cross-section
        x_cross = torch.linspace(-60, 60, 100)
        y_cross = torch.linspace(-60, 0, 60)
        X, Y = torch.meshgrid(x_cross, y_cross, indexing='ij')
        
        # Calculate resistivity at each point
        resistivity_map = torch.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = torch.tensor([X[i, j], Y[i, j]])
                conductivity = conductivity_field(point)
                resistivity_map[i, j] = 1.0 / conductivity
        
        # Create contour plot
        contour = ax2.contourf(X.numpy(), Y.numpy(), resistivity_map.numpy(), 
                              levels=20, cmap='viridis')
        ax2.set_xlabel('Distance (m)')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('Subsurface Resistivity Distribution')
        
        # Mark current injection points
        ax2.plot([-10, 10], [0, 0], 'r^', markersize=8, label='Current electrodes')
        
        # Mark measurement electrodes
        ax2.plot(x_positions, np.zeros_like(x_positions), 'wo', markersize=3, 
                alpha=0.7, label='Measurement electrodes')
        
        ax2.legend()
        plt.colorbar(contour, ax=ax2, label='Resistivity (Ω⋅m)')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('dcr_survey_results.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'dcr_survey_results.png'")
        
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Continuing without plots...")

def main():
    """
    Main function to run DCR survey test and display results.
    """
    try:
        measurement_positions, measured_voltages = run_dcr_survey_simulation()
        plot_dcr_survey_results(measurement_positions, measured_voltages, save_plot=True)
        
        return True
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return False

if __name__ == "__main__":
    # Run main DCR survey simulation
    success = main()
