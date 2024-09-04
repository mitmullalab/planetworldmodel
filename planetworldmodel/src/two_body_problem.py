import numpy as np
from pydantic import BaseModel, Field


class TwoBodyProblem(BaseModel):
    mass_1: float
    mass_2: float
    r_0_magnitude: float = Field(
        ..., description="Magnitude of the initial relative position vector"
    )
    r_0_angle: float = Field(
        ...,
        description="Angle in the x-y plane of the initial relative position vector",
    )
    v_0_magnitude: float = Field(
        ..., description="Magnitude of the initial relative velocity vector"
    )
    v_0_angle: float = Field(
        ...,
        description="Angle in the x-y plane of the initial relative velocity vector",
    )
    gravitational_constant: float = 6.67430e-11


def compute_position(nu: float, a: float, e: float) -> np.ndarray:
    """Compute the position vector in the orbital plane.
    Assumes the entire orbit is in the x-y plane.

    Args:
        nu: True anomaly
        a: Semi-major axis
        e: Eccentricity

    Returns:
        Position vector in the orbital plane.
    """
    if e < 1:  # Elliptical orbit
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
    else:  # Hyperbolic orbit
        r = a * (e**2 - 1) / (1 + e * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    z = 0
    return np.array([x, y, z])


def generate_trajectories(
    problem: TwoBodyProblem, num_points: int = 1_0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate the trajectories of the two objects in the two-body problem.
    Assumes the entire orbit is in the x-y plane.

    Args:
        problem: TwoBodyProblem instance that contains the problem parameters.
        num_points: Number of points to generate along the orbit.

    Returns:
        Tuple of two numpy arrays representing the trajectories of the two objects.
        The third element is the eccentricity of the orbit.
    """
    # Total mass and reduced mass
    mass_tot = problem.mass_1 + problem.mass_2
    mass_red = problem.gravitational_constant * mass_tot

    # Compute eccentricity vector
    r_0 = np.array(
        [
            problem.r_0_magnitude * np.cos(problem.r_0_angle),
            problem.r_0_magnitude * np.sin(problem.r_0_angle),
            0,
        ]
    )
    v_0 = np.array(
        [
            problem.v_0_magnitude * np.cos(problem.v_0_angle),
            problem.v_0_magnitude * np.sin(problem.v_0_angle),
            0,
        ]
    )
    h_0 = np.cross(r_0, v_0)  # Specific angular momentum
    r, v = np.linalg.norm(r_0), np.linalg.norm(v_0)
    e_vec = np.cross(v_0, h_0) / mass_red - r_0 / r
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    a = 1 / (2 / r - v**2 / mass_red)

    # Calculate orbit
    if e < 1:  # Elliptical orbit
        nu_vals = np.linspace(0, 2 * np.pi, num_points)
    else:  # Hyperbolic orbit
        nu_max = np.arccos(-1 / e)
        nu_vals = np.linspace(-nu_max, nu_max, num_points)
    orbit_rel = np.array([compute_position(nu, a, e) for nu in nu_vals])

    # Calculate the two objects' orbits
    orbit_1 = -orbit_rel * (problem.mass_2 / mass_tot)
    orbit_2 = orbit_rel * (problem.mass_1 / mass_tot)
    return orbit_1, orbit_2, e
