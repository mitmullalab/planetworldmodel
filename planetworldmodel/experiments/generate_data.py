import argparse

import numpy as np
import tqdm

from planetworldmodel import generate_trajectories, random_two_body_problem
from planetworldmodel.setting import DATA_DIR


def main(args):
    count = 0
    pbar = tqdm.tqdm(
        total=len(args.eccentricities) * args.num_trajectories_per_eccentricity
    )
    traj_min, traj_max = np.inf, -np.inf
    train, val = [], []
    for e in args.eccentricities:
        for _ in range(args.num_trajectories_per_eccentricity):
            for dataset in (train, val):
                problem = random_two_body_problem(target_eccentricity=e, seed=count)
                traj_1, traj_2, _ = generate_trajectories(
                    problem, args.num_points, args.dt
                )
                traj = np.concatenate((traj_1[:, :2], traj_2[:, :2]), axis=1)
                traj_min, traj_max = (
                    min(traj_min, traj.min()),
                    max(traj_max, traj.max()),
                )
                dataset.append(traj)
                count += 1
            pbar.update(1)
    pbar.close()
    print(f"Trajectory min: {traj_min}")
    print(f"Trajectory max: {traj_max}")

    # Save as a numpy file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "two_body_problem_train.npy", np.array(train))
    np.save(DATA_DIR / "two_body_problem_val.npy", np.array(val))


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

    args = parser.parse_args()
    main(args)
