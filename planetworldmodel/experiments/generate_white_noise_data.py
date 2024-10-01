import argparse

import numpy as np
import tqdm

from planetworldmodel import (
    generate_trajectories,
    generate_trajectory_with_heavier_fixed,
    random_two_body_problem,
)
from planetworldmodel.setting import DATA_DIR


def signed_log(x):
    """Compute the signed log of x."""
    return np.sign(x) * np.log1p(np.abs(x))


def generate_data(
    num_points: int,
    num_trajectories_per_eccentricity: int,
    eccentricities: list[float],
    dt: float,
    obs_variance: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pbar = tqdm.tqdm(total=len(eccentricities) * num_trajectories_per_eccentricity)
    traj_min, traj_max = np.inf, -np.inf
    obs_ls, noise_ls, indx_ls = [], [], []
    for e in eccentricities:
        for _ in range(num_trajectories_per_eccentricity):
            problem = random_two_body_problem(target_eccentricity=e, seed=seed)
            seed += 1
            obs, state = generate_trajectory_with_heavier_fixed(
                problem, num_points, dt, obs_variance, rng=seed
            )
            seed += 1
            traj = obs.trajectory1
            obs_ls.append(traj)
            num_points = len(traj)
            rng = np.random.default_rng(seed)
            seed += 1
            white_noise = rng.normal(0, scale=2.0, size=(num_points, 1))
            noise_ls.append(white_noise)
            
            # Generate random sequence index
            rng = np.random.default_rng(seed)
            seed += 1
            indx = rng.integers(0, num_points)
            indx_ls.append(indx)
            
            traj_min, traj_max = (
                min(traj_min, np.array(traj).min()),
                max(traj_max, np.array(traj).max()),
            )
            pbar.update(1)
    pbar.close()
    print(f"Trajectory min: {traj_min}")
    print(f"Trajectory max: {traj_max}")
    return np.array(obs_ls), np.array(noise_ls), np.array(indx_ls), seed


def main(args):
    seed = 0
    obs_tr, noise_tr, indx_tr, seed = generate_data(
        args.num_points,
        args.num_trajectories_per_eccentricity,
        args.eccentricities,
        args.dt,
        args.obs_variance,
        seed,
    )
    
    obs_val, noise_val, indx_val, _ = generate_data(
        args.num_points,
        args.num_trajectories_per_eccentricity,
        args.eccentricities,
        args.dt,
        args.obs_variance,
        seed,
    )

    # Save as a numpy file
    data_dir = DATA_DIR / "white_noise_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "obs_train_heavier_fixed.npy", obs_tr)
    np.save(data_dir / "indx_train_heavier_fixed.npy", indx_tr)
    np.save(data_dir / "noise_train_heavier_fixed.npy", noise_tr)
    np.save(data_dir / "obs_val_heavier_fixed.npy", obs_val)
    np.save(data_dir / "indx_val_heavier_fixed.npy", indx_val)
    np.save(data_dir / "noise_val_heavier_fixed.npy", noise_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data for the two-body problem."
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1_000,
        help="Number of points to generate along the orbit.",
    )
    parser.add_argument(
        "--num_trajectories_per_eccentricity",
        type=int,
        default=100,
        help="Number of trajectories to generate per eccentricity.",
    )
    parser.add_argument(
        "--eccentricities",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=10,  # 10 seconds
        help="Time step in seconds between each point.",
    )
    parser.add_argument(
        "--obs_variance",
        type=float,
        default=0.0,
        help="Variance of the observation noise.",
    )

    args = parser.parse_args()
    main(args)
