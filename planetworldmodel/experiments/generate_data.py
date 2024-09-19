import argparse

import numpy as np
import tqdm

from planetworldmodel import generate_trajectories, random_two_body_problem
from planetworldmodel.setting import DATA_DIR


def main(args):
    seed = 0
    pbar = tqdm.tqdm(
        total=len(args.eccentricities) * args.num_trajectories_per_eccentricity
    )
    traj_min, traj_max = np.inf, -np.inf
    train, val = [], []
    for e in args.eccentricities:
        for _ in range(args.num_trajectories_per_eccentricity):
            for dataset in (train, val):
                problem = random_two_body_problem(target_eccentricity=e, seed=seed)
                seed += 1
                traj_1, traj_2, _ = generate_trajectories(
                    problem, args.num_points, args.dt, args.obs_variance, seed
                )
                seed += 1
                traj = np.concatenate((traj_1[:, :2], traj_2[:, :2]), axis=1)
                traj_min, traj_max = (
                    min(traj_min, traj.min()),
                    max(traj_max, traj.max()),
                )
                dataset.append(traj)
            pbar.update(1)
    pbar.close()
    print(f"Trajectory min: {traj_min}")
    print(f"Trajectory max: {traj_max}")

    # Save as a numpy file
    data_dir = DATA_DIR / f"obs_var_{args.obs_variance:.5f}"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "two_body_problem_train.npy", np.array(train))
    np.save(data_dir / "two_body_problem_val.npy", np.array(val))


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
        default=50_000,
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
