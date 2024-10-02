import argparse

import numpy as np
import tqdm

from planetworldmodel import (
    generate_trajectory_with_heavier_fixed,
    random_two_body_problem,
)
from planetworldmodel.setting import DATA_DIR


def compute_scaling_factor(force, target_max=6):
    """Compute a single scaling factor for both x and y components."""
    max_abs_force = np.max(np.abs(force))
    return target_max / max_abs_force


def scale_force(force, scaling_factor):
    """Scale force using the pre-computed scaling factor."""
    return force * scaling_factor


def generate_data(
    num_points: int,
    num_trajectories_per_eccentricity: int,
    eccentricities: list[float],
    dt: float,
    fix_heavier_object_across_sequences: bool,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pbar = tqdm.tqdm(total=len(eccentricities) * num_trajectories_per_eccentricity)
    traj_min, traj_max = np.inf, -np.inf
    obs_ls, force_ls, heavier_coord_ls, m_light_ls, m_heavy_ls = [], [], [], [], []
    heavier_coord = None
    if fix_heavier_object_across_sequences:
        heavier_coord = np.array([0., 0.])  # Fix the heavier object at the origin
    for e in eccentricities:
        for _ in range(num_trajectories_per_eccentricity):
            problem = random_two_body_problem(target_eccentricity=e, seed=seed)
            seed += 1
            obs, state = generate_trajectory_with_heavier_fixed(
                problem, num_points, dt, rng=seed, heavier_coord=heavier_coord,
            )
            traj = obs.trajectory1
            obs_ls.append(traj)
            state_dict = state.model_dump()
            force_vec = np.array(state_dict["force_vector"])
            force_ls.append(force_vec)
            num_points = len(force_vec)
            traj_heavy = np.tile(state_dict["trajectory_heavy"], (num_points, 1))
            heavier_coord_ls.append(traj_heavy)
            m_light_ls.append(np.tile(state_dict["m_light"], (num_points, 1)))
            m_heavy_ls.append(np.tile(state_dict["m_heavy"], (num_points, 1)))
            traj_min, traj_max = (
                min(traj_min, np.array(traj).min()),
                max(traj_max, np.array(traj).max()),
            )
            seed += 1
            pbar.update(1)
    pbar.close()
    print(f"Trajectory min: {traj_min}")
    print(f"Trajectory max: {traj_max}")
    return (
        np.array(obs_ls), np.array(force_ls), np.array(heavier_coord_ls),
        np.array(m_light_ls), np.array(m_heavy_ls), seed,
    )


def main(args):
    seed = 0
    obs_tr, state_tr, heavier_tr, m_light_tr, m_heavy_tr, seed = generate_data(
        args.num_points,
        args.num_train_trajectories_per_eccentricity,
        args.eccentricities,
        args.dt,
        args.fix_heavier_object_across_sequences,
        seed,
    )
    force_x_tr = state_tr[:, :, 0]
    force_y_tr = state_tr[:, :, 1]
    
    # Compute scaling factors from training data
    scaling_factor_x = compute_scaling_factor(force_x_tr)
    scaling_factor_y = compute_scaling_factor(force_y_tr)

    # Scale the training force vectors
    scaled_force_x_tr = scale_force(force_x_tr, scaling_factor_x)
    scaled_force_y_tr = scale_force(force_y_tr, scaling_factor_y)
    force_vec_tr = np.stack([scaled_force_x_tr, scaled_force_y_tr], axis=-1)
    
    obs_val, state_val, heavier_val, m_light_val, m_heavy_val, seed = generate_data(
        args.num_points,
        args.num_val_trajectories_per_eccentricity,
        args.eccentricities,
        args.dt,
        args.fix_heavier_object_across_sequences,
        seed,
    )
    
    # Normalize the validation force vectors using training set parameters
    force_x_val = state_val[:, :, 0]
    force_y_val = state_val[:, :, 1]
    scaled_force_x_val = scale_force(force_x_val, scaling_factor_x)
    scaled_force_y_val = scale_force(force_y_val, scaling_factor_y)
    force_vec_val = np.stack([scaled_force_x_val, scaled_force_y_val], axis=-1)
    
    obs_test, state_test, heavier_test, m_light_test, m_heavy_test, _ = generate_data(
        args.num_points,
        args.num_test_trajectories_per_eccentricity,
        args.eccentricities,
        args.dt,
        args.fix_heavier_object_across_sequences,
        seed,
    )
    
    # Normalize the test force vectors using training set parameters
    force_x_test = state_test[:, :, 0]
    force_y_test = state_test[:, :, 1]
    scaled_force_x_test = scale_force(force_x_test, scaling_factor_x)
    scaled_force_y_test = scale_force(force_y_test, scaling_factor_y)
    force_vec_test = np.stack([scaled_force_x_test, scaled_force_y_test], axis=-1)

    # Save as a numpy file
    data_dir = DATA_DIR / f"force_vectors"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_name = "_heavier_fixed" if args.fix_heavier_object_across_sequences else ""
    np.save(data_dir / f"obs_train{file_name}.npy", obs_tr)
    np.save(data_dir / f"obs_val{file_name}.npy", obs_val)
    np.save(data_dir / f"obs_test{file_name}.npy", obs_test)
    np.save(data_dir / f"force_train{file_name}.npy", force_vec_tr)
    np.save(data_dir / f"force_val{file_name}.npy", force_vec_val)
    np.save(data_dir / f"force_test{file_name}.npy", force_vec_test)
    np.save(data_dir / f"heavier_train{file_name}.npy", heavier_tr)
    np.save(data_dir / f"heavier_val{file_name}.npy", heavier_val)
    np.save(data_dir / f"heavier_test{file_name}.npy", heavier_test)
    np.save(data_dir / f"m_light_train{file_name}.npy", m_light_tr)
    np.save(data_dir / f"m_light_val{file_name}.npy", m_light_val)
    np.save(data_dir / f"m_light_test{file_name}.npy", m_light_test)
    np.save(data_dir / f"m_heavy_train{file_name}.npy", m_heavy_tr)
    np.save(data_dir / f"m_heavy_val{file_name}.npy", m_heavy_val)
    np.save(data_dir / f"m_heavy_test{file_name}.npy", m_heavy_test)


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
        "--num_train_trajectories_per_eccentricity",
        type=int,
        default=10_000,
        help="Number of trajectories to generate per eccentricity.",
    )
    parser.add_argument(
        "--num_val_trajectories_per_eccentricity",
        type=int,
        default=100,
        help="Number of trajectories to generate per eccentricity.",
    )
    parser.add_argument(
        "--num_test_trajectories_per_eccentricity",
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
        "--fix_heavier_object_across_sequences",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
