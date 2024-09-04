from planetworldmodel import random_two_body_problem, generate_trajectories

import pytest


@pytest.mark.parametrize("target_eccentricity", [0.2, 0.5, 0.9, 1.0, 1.1, 10.0])
def test_random_two_body_problem(target_eccentricity, tol=1e-3):
    two_body_problem = random_two_body_problem(target_eccentricity)
    _, _, eccentricity = generate_trajectories(two_body_problem)
    assert abs(eccentricity - target_eccentricity) <= tol
