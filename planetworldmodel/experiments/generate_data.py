import argparse

import numpy as np
import tqdm

from planetworldmodel import (
    generate_trajectories,
    generate_trajectory_with_heavier_fixed,
    random_two_body_problem,
)
from planetworldmodel.setting import DATA_DIR


def generate_data(
    num_points: int,
    num_trajectories_per_eccentricity: int,
    eccentricities: list[float],
    dt: float,
    obs_variance: float,
    fix_heavier_object: bool,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    pbar = tqdm.tqdm(total=len(eccentricities) * num_trajectories_per_eccentricity)
    traj_min, traj_max = np.inf, -np.inf
    obs_ls, state_ls = [], []
    print(f"seed: {seed}")
    for e in eccentricities:
        for _ in range(num_trajectories_per_eccentricity):
            curr_state = []
            problem = random_two_body_problem(target_eccentricity=e, seed=seed)
            seed += 1
            if fix_heavier_object:
                obs, state = generate_trajectory_with_heavier_fixed(
                    problem, num_points, dt, obs_variance, rng=seed
                )
                traj = obs.trajectory1
                obs_ls.append(traj)
                state_dict = state.model_dump()
                traj_light = np.array(state_dict["trajectory_light"])
                num_points = len(traj_light)
                traj_heavy = np.tile(state_dict["trajectory_heavy"], (num_points, 1))
                v_relative = np.array(state_dict["relative_velocity"])
                m_light = np.tile(state_dict["m_light"], (num_points, 1))
                m_heavy = np.tile(state_dict["m_heavy"], (num_points, 1))
                energy = np.tile(state_dict["energy"], (num_points, 1))
                angular_momentum = np.tile(
                    state_dict["angular_momentum"], (num_points, 1)
                )
                center_of_mass = np.array(state_dict["center_of_mass"])
                curr_state = np.concatenate(
                    [
                        traj_light,
                        traj_heavy,
                        v_relative,
                        m_light,
                        m_heavy,
                        energy,
                        angular_momentum,
                        center_of_mass,
                    ],
                    axis=1,
                )
                state_ls.append(curr_state)
            else:
                traj_1, traj_2, *_ = generate_trajectories(
                    problem, num_points, dt, obs_variance, rng=seed
                )
                traj = np.concatenate((traj_1, traj_2), axis=1)
                obs_ls.append(traj)
            traj_min, traj_max = (
                min(traj_min, np.array(traj).min()),
                max(traj_max, np.array(traj).max()),
            )
            seed += 1
            pbar.update(1)
    pbar.close()
    print(f"Trajectory min: {traj_min}")
    print(f"Trajectory max: {traj_max}")
    return np.array(obs_ls), np.array(state_ls), seed


def main(args):
    seed = int(args.iteration * 5e7)
    # obs_tr, state_tr, seed = generate_data(
    #     args.num_points,
    #     args.num_train_trajectories_per_eccentricity,
    #     args.eccentricities,
    #     args.dt,
    #     args.obs_variance,
    #     args.fix_heavier_object,
    #     seed,
    # )
    # obs_val, state_val, _ = generate_data(
    #     args.num_points,
    #     args.num_val_trajectories_per_eccentricity,
    #     args.eccentricities,
    #     args.dt,
    #     args.obs_variance,
    #     args.fix_heavier_object,
    #     seed,
    # )
    obs_test, state_test, _ = generate_data(
        args.num_points,
        args.num_val_trajectories_per_eccentricity,
        args.eccentricities,
        args.dt,
        args.obs_variance,
        args.fix_heavier_object,
        seed,
    )
    

    # Save as a numpy file
    # data_dir = DATA_DIR / f"obs_var_{args.obs_variance:.5f}"
    # data_dir = DATA_DIR / "transfer_data"
    data_dir = DATA_DIR / "two_body"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.fix_heavier_object:
        # np.save(data_dir / f"obs_train_heavier_fixed_{args.iteration}.npy", obs_tr)
        # np.save(data_dir / f"obs_val_heavier_fixed.npy", obs_val)
        # np.save(data_dir / f"state_train_heavier_fixed_{args.iteration}.npy", state_tr)
        # np.save(data_dir / f"state_val_heavier_fixed.npy", state_val)
        np.save(data_dir / f"obs_test_heavier_fixed.npy", obs_test)
        np.save(data_dir / f"state_test_heavier_fixed.npy", state_test)
    else:
        ...
        # np.save(data_dir / f"two_body_problem_train_{args.iteration}.npy", np.array(obs_tr))
        # np.save(data_dir / f"two_body_problem_val_{args.iteration}.npy", np.array(obs_val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data for the two-body problem."
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=200,
        help="Number of points to generate along the orbit.",
    )
    parser.add_argument(
        "--num_train_trajectories_per_eccentricity",
        type=int,
        default=10_000,
        help="Number of trajectories to generate per eccentricity.",
    )
    parser.add_argument(
        "--num_val_trajectories_per_eccentricity",
        type=int,
        default=1_000,
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
        default=300,  # 5 minutes
        help="Time step in seconds between each point.",
    )
    parser.add_argument(
        "--obs_variance",
        type=float,
        default=0.0,
        help="Variance of the observation noise.",
    )
    parser.add_argument(
        "--fix_heavier_object",
        action="store_true",
        help="Fix the heavier object at some random coordinate.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    main(args)
