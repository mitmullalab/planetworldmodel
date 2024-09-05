import numpy as np
from pydantic import BaseModel, Field

float_type = float | np.floating


class TwoBodyProblem(BaseModel):
    mass_1: float_type
    mass_2: float_type
    r_0_magnitude: float_type = Field(
        ..., description="Magnitude of the initial relative position vector"
    )
    r_0_angle: float_type = Field(
        ...,
        description="Angle in the x-y plane of the initial relative position vector",
    )
    v_0_magnitude: float_type = Field(
        ..., description="Magnitude of the initial relative velocity vector"
    )
    v_0_angle: float_type = Field(
        ...,
        description="Angle in the x-y plane of the initial relative velocity vector",
    )
    gravitational_constant: float_type = 6.67430e-11


def random_two_body_problem(
    target_eccentricity: float_type, seed: int = 0, g_constant: float_type = 6.67430e-11
) -> TwoBodyProblem:
    """
    Generate a random TwoBodyProblem instance with a specified target eccentricity.

    Args:
        target_eccentricity: The desired eccentricity of the orbit.
        seed: Random number generator seed.
        g_constant: Gravitational constant.

    Returns:
        A TwoBodyProblem instance with the specified eccentricity.
    """
    rng = np.random.default_rng(seed)

    # Generate random masses (1e24 to 1e30 kg, log-uniform)
    mass_1 = 10 ** rng.uniform(24, 30)
    mass_2 = 10 ** rng.uniform(24, 30)
    mass_tot = mass_1 + mass_2
    mu = g_constant * mass_tot

    # Generate random initial position (1e9 to 1e12 m, log-uniform)
    r_0_magnitude = 10 ** rng.uniform(9, 12)
    r_0_angle = rng.uniform(0, 2 * np.pi)

    # Calculate semi-major axis and velocity magnitude
    if target_eccentricity < 1:  # Elliptical orbit
        a = r_0_magnitude / (1 - target_eccentricity)
        v_0_magnitude = np.sqrt(mu * (2 / r_0_magnitude - 1 / a))
    elif target_eccentricity == 1:  # Parabolic orbit
        v_0_magnitude = np.sqrt(2 * mu / r_0_magnitude)
    else:  # Hyperbolic orbit
        a = r_0_magnitude / (target_eccentricity - 1)
        v_0_magnitude = np.sqrt(mu * (2 / r_0_magnitude + 1 / np.abs(a)))

    # Calculate the angle of the velocity vector
    # We want the velocity to be perpendicular to the eccentricity vector
    # e = ((v^2 - μ/r)r - (r·v)v) / μ
    # For simplicity, let's choose v perpendicular to r
    v_0_angle = r_0_angle + np.pi / 2

    # Adjust v_0_magnitude to achieve the desired eccentricity
    v_squared = mu * (1 + target_eccentricity) / r_0_magnitude
    v_0_magnitude = np.sqrt(v_squared)

    return TwoBodyProblem(
        mass_1=mass_1,
        mass_2=mass_2,
        r_0_magnitude=r_0_magnitude,
        r_0_angle=r_0_angle,
        v_0_magnitude=v_0_magnitude,
        v_0_angle=v_0_angle,
        gravitational_constant=g_constant,
    )


def compute_parabolic_position(
    u: float_type, h: np.ndarray, mu: float_type
) -> np.ndarray:
    """Compute the position vector for a parabolic orbit.
    Assumes the entire orbit is in the x-y plane.

    Args:
        u: Anomaly angle
        h: Specific angular momentum
        mu: Gravitational parameter

    Returns:
        Position vector in the orbital plane.
    """
    p = np.linalg.norm(h) ** 2 / mu  # parameter of the parabola
    r = p / 2 * (1 + np.cos(u))
    x = r * np.cos(u)
    y = r * np.sin(u)
    return np.array([x, y, 0])


def compute_position(nu: float_type, a: float_type, e: float_type) -> np.ndarray:
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
    problem: TwoBodyProblem,
    num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, float_type]:
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

    print(f"mass_tot: {mass_tot}, mass_red: {mass_red}")
    print(f"r_0, v_0: {r_0}, {v_0}")
    print(f"h_0: {h_0}")
    print(f"e_vec: {e_vec}, e: {e}")
    print(f"a: {a}")

    # Calculate orbit
    if e == 1:  # Paraoblic orbit
        u_vals = np.linspace(-np.pi, np.pi, num_points)
        orbit_rel = np.array(
            [compute_parabolic_position(u, h_0, mass_red) for u in u_vals]
        )
    else:
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
