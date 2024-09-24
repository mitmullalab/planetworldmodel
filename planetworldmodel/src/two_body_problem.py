import numpy as np
from numpy.random import Generator
from pydantic import BaseModel, Field
from scipy.optimize import newton

float_type = float | np.floating


PARABOLIC_TOLERANCE = 1e-5  # A small tolerance for checking if e is close to 1


class TwoBodyProblem(BaseModel):
    model_config: dict = {
        "arbitrary_types_allowed": True,
    }

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


def is_nearly_parabolic(e: float_type) -> bool:
    """Check if the orbit is nearly parabolic."""
    return np.abs(e - 1) < PARABOLIC_TOLERANCE


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

    # Generate random masses (1e2 to 1e5 kg, log-uniform)
    mass_1 = 10 ** rng.uniform(2, 5)
    mass_2 = 10 ** rng.uniform(2, 5)
    mass_tot = mass_1 + mass_2
    mu = g_constant * mass_tot

    # Generate random initial position (1 to 10 m, uniform)
    r_0_magnitude = rng.uniform(1, 10)
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


def calculate_orbital_parameters(
    r: float_type, v: float_type, mass_red: float_type
) -> tuple[float_type, float_type]:
    """
    Calculate orbital parameters (semi-major axis/parabola parameter and eccentricity).

    Args:
        r: Magnitude of the position vector
        v: Magnitude of the velocity vector
        mass_red: Reduced mass of the system (G * (m1 + m2))

    Returns:
        Tuple of (a, e) where:
        a: Semi-major axis for elliptic/hyperbolic orbits
            (or parabola parameter for parabolic orbits)
        e: Eccentricity
    """
    # Specific orbital energy
    energy = v**2 / 2 - mass_red / r
    h = r * v  # Specific angular momentum

    if np.isclose(energy, 0, atol=1e-8):  # Parabolic orbit
        # For parabolic orbits, we calculate the parameter p
        p = h**2 / mass_red
        return p / 2, 1.0  # Return p/2 as 'a' and e=1
    # For elliptic and hyperbolic orbits
    a = -mass_red / (2 * energy)
    e = np.sqrt(1 + 2 * energy * h**2 / mass_red**2)
    return a, e


def kepler_equation_elliptic(E: float_type, M: float_type, e: float_type) -> float_type:
    """Kepler's Equation for elliptical orbits."""
    return E - e * np.sin(E) - M


def kepler_equation_hyperbolic(
    H: float_type, M: float_type, e: float_type
) -> float_type:
    """Kepler's Equation for hyperbolic orbits."""
    return e * np.sinh(H) - H - M


def solve_kepler_equation(M: float_type, e: float_type) -> float_type:
    """Solve Kepler's Equation for elliptic and hyperbolic orbits."""
    if is_nearly_parabolic(e):  # Parabolic
        return M
    if e < 1:  # Elliptic
        return newton(lambda E: kepler_equation_elliptic(E, M, e), M)
    # Hyperbolic
    return newton(lambda H: kepler_equation_hyperbolic(H, M, e), np.asinh(M / e))


def true_anomaly_from_anomaly(anomaly: float_type, e: float_type) -> float_type:
    """Convert eccentric/hyperbolic anomaly to true anomaly."""
    if is_nearly_parabolic(e):  # Parabolic
        return anomaly  # For parabolic orbits, we directly use the "anomaly" as true anomaly
    if e < 1:  # Elliptic
        return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(anomaly / 2))
    # Hyperbolic
    return 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(anomaly / 2))


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
    if is_nearly_parabolic(e):
        p = 2 * a
        r = p / (1 + np.cos(nu))
    elif e < 1:  # Elliptical orbit
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
    else:  # Hyperbolic orbit
        r = a * (e**2 - 1) / (1 + e * np.cos(nu))
    return np.array([r * np.cos(nu), r * np.sin(nu), 0])


def generate_trajectories(
    problem: TwoBodyProblem,
    num_points: int = 1_000,
    dt: float_type = 10,  # 10 second
    obs_variance: float_type = 0.0,
    rng: int | Generator = 0,
) -> tuple[np.ndarray, np.ndarray, float_type]:
    """Generate the trajectories of the two objects in the two-body problem.
    Assumes the entire orbit is in the x-y plane.

    Args:
        problem: TwoBodyProblem instance that contains the problem parameters.
        num_points: Number of points to generate along the orbit.
        dt: Time step in seconds between each point.
        obs_variance: Variance of the observation noise.
        rng: Random number generator or seed.

    Returns:
        Tuple of two numpy arrays representing the trajectories of the two objects.
        The third element is the eccentricity of the orbit.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

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
    r, v = np.linalg.norm(r_0), np.linalg.norm(v_0)

    # Calculate orbital parameters
    a, e = calculate_orbital_parameters(r, v, mass_red)

    # Generate equally spaced times
    t = np.arange(num_points) * dt

    # Calculate orbit
    if is_nearly_parabolic(e):
        p = 2 * a
        n = np.sqrt(mass_red / (2 * p**3))
    elif e < 1:  # Elliptical orbit
        n = np.sqrt(mass_red / a**3)  # Mean motion
    else:  # Hyperbolic orbit
        n = np.sqrt(-mass_red / a**3)
    M_vals = n * t

    if is_nearly_parabolic(e):
        nu = 2 * np.arctan(M_vals)
    else:
        # Solve Kepler's equation and compute true anomaly
        nu = np.array(
            [
                true_anomaly_from_anomaly(solve_kepler_equation(Mi, e), e)
                for Mi in M_vals
            ]
        )

    # Calculate orbit
    orbit_rel = np.array([compute_position(nui, a, e) for nui in nu])

    # Calculate the two objects' orbits
    orbit_1 = -orbit_rel * (problem.mass_2 / mass_tot)
    orbit_2 = orbit_rel * (problem.mass_1 / mass_tot)

    # Add observation noise
    orbit_1 += rng.normal(0, np.sqrt(obs_variance), size=orbit_1.shape)
    orbit_2 += rng.normal(0, np.sqrt(obs_variance), size=orbit_2.shape)
    return orbit_1, orbit_2, e


def compute_relative_orbit(
    heavier_orbit: np.ndarray, lighter_orbit: np.ndarray, heavier_coord: np.ndarray
) -> np.ndarray:
    """Compute the lighter object's orbit relative to the fixed heavier object.

    Args:
        heavier_orbit: The original orbit of the heavier object
        lighter_orbit: The original orbit of the lighter object
        heavier_coord: The fixed coordinates of the heavier object

    Returns:
        The adjusted orbit of the lighter object relative to the fixed heavier object
    """
    # Calculate the offset of the heavier object from its fixed position
    offset = heavier_orbit - heavier_coord

    # Adjust the lighter object's orbit by subtracting this offset
    relative_lighter_orbit = lighter_orbit - offset
    return relative_lighter_orbit


def generate_trajectory_with_heavier_fixed(
    problem: TwoBodyProblem,
    num_points: int = 1_000,
    dt: float_type = 10,  # 10 second
    obs_variance: float_type = 0.0,
    rng: int | Generator = 0,
) -> tuple[np.ndarray, np.ndarray, float_type]:
    """Generate a trajectory with the heavier object fixed at some random coordinate.

    Args:
        problem: TwoBodyProblem instance that contains the problem parameters.
        num_points: Number of points to generate along the orbit.
        dt: Time step in seconds between each point.
        obs_variance: Variance of the observation noise.
        rng: Random number generator or seed.

    Returns:
        Tuple of the fixed coordinates of the heavier object, the relative orbit
        of the lighter object, and the eccentricity of the orbit.
    """

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    orbit_1, orbit_2, e = generate_trajectories(
        problem, num_points, dt, obs_variance, rng
    )

    heavier_orbit, lighter_orbit = (
        (orbit_1, orbit_2) if problem.mass_1 > problem.mass_2 else (orbit_2, orbit_1)
    )

    # Randomly sample the heavier object's coordinates
    heavier_coord = rng.choice(heavier_orbit, size=1, axis=0).squeeze()

    # Compute the lighter object's relative trajectory
    relative_lighter_orbit = compute_relative_orbit(
        heavier_orbit, lighter_orbit, heavier_coord
    )
    return heavier_coord, relative_lighter_orbit, e
