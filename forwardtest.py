import torch
import zombie
import numpy as np  
from typing import Callable


def create_grid_points_2d(grid_size: tuple[float, float], grid_res: tuple[float, float]) -> torch.Tensor:
    """
    CReate a uniform grid of points in the specified range, evenly spaced according to the resolution.
    Args:
        grid_size (tuple[float, float]): The size of the grid in the x and y dimensions.
        grid_res (tuple[float, float]): The resolution of the grid in the x and y dimensions.
    """

    x = torch.arange(0, grid_size[0], grid_res[0])
    y = torch.arange(0, grid_size[1], grid_res[1])
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)



def build_boundary_function_2d(surface_topo: Callable[[float], float], boundary_size: tuple[float, float]) -> zombie.core.geometric_queries_2d:
    """
    Build an implict description of the boundary function for the surface topology.
    The top boundary is defined as a an arbitrary function of the x-coordinate, the side and bottom boundaries are defined as just coonstant lines.
    By convention the origin of the coordinate system is at the bottom left corner of the domain.
    Args:
        surface_topo (Callable[[float], float]): A function that takes an x-coordinate and returns the y-coordinate of the surface topology.
        boundary_size (tuple[float, float]): The size of the boundary in the x and y dimensions.
    Returns:
        zombie.core.geometric_queries_2d: A geometric query object that can be used to check if a point is inside the domain defined by the surface topology and boundary size.
    """

    def inside_domain(p: np.ndarray) -> bool:
        x, y = p
        return 0 <= x <= boundary_size[0] and 0 <= y <= boundary_size[1] and y >= surface_topo(x)
    
    geometric_queries = zombie.core.geometric_queries_2d(
        inside_domain,
        domain_min=np.array([0.0, 0.0]),
        domain_max=np.array(boundary_size),
    )

    # We will also define boundary conditions here since top boundary is Neumann and the sides are Dirichlet
    def is_reflecting_boundary(p: np.ndarray) -> bool:
        x, y = p
        return (y - surface_topo(x)) < 1e-6 # top boundary is Neumann

    return geometric_queries, is_reflecting_boundary

def build_electrode_sources_2d(electrode_positions: list[tuple[float, float]], electrode_values: list[float]) -> callable[np.ndarray, float]:
    """
    Build a set of electrode sources for the forward problem.
    Args:
        electrode_positions (list[tuple[float, float]]): A list of tuples representing the positions of the electrodes in the x and y dimensions.
        electrode_values (list[float]): A list of values for each electrode.
    Returns:
        callable[np.ndarray, bool]: A function that takes a point in the domain and returns its current value.
    """
    def electrode_source(p: np.ndarray) -> float:
        x, y = p
        for (ex, ey), value in zip(electrode_positions, electrode_values):
            if np.isclose(x, ex) and np.isclose(y, ey):
                return value
        return 0.0  # Default value if not an electrode position

    return electrode_source
    
    

def build_pde() -> zombie.core.pde_2d:
    """
    Build a PDE object for the forward problem.
    The PDE is defined as a Laplace equation with a source term.
    Returns:
        zombie.core.pde_2d: A PDE object that can be used to solve the forward problem.
    """
    pde = zombie.core.pde_float_2d()


if __name__ == "__main__":
    # Example usage here, for our purpose we will simulate a simple 2D solver with a sinusoidal surface topology. and an conductive ellipsoid body
    print("Running forward test...")