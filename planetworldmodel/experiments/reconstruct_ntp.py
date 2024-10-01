import logging

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from planetworldmodel import (
    TransformerConfig,
    load_config,
)
from planetworldmodel.setting import GEN_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reconstruct_next_position_from_state(
    predicted_state: jax.Array, curr_position: jax.Array, dt=10, G=6.67430e-11
):
    """Reconstruct the trajectories from the state predictions.

    Args:
        predicted_state: Predicted state of the current time step.
        curr_position: Input position of the current time step.
        dt: Time step.
        G: Gravitational constant.

    Returns:
        Reconstructed next-step position.
    """
    x_h, y_h = predicted_state[2:4]
    vx, vy = predicted_state[4:6]
    x_l, y_l = curr_position
    log_m = predicted_state[-1]
    m = jnp.exp(log_m)

    # Calculate relative position
    rx = x_l - x_h
    ry = y_l - y_h
    r = jnp.sqrt(rx**2 + ry**2)

    # Calculate gravitational acceleration (only affects lighter object)
    a = G * m / (r**2)
    a_x = -a * rx / r
    a_y = -a * ry / r

    # Update velocity (relative to fixed heavy object)
    vx_new = vx + a_x * dt
    vy_new = vy + a_y * dt

    # Update position (relative to fixed heavy object)
    rx_new = rx + vx_new * dt + 0.5 * a_x * dt**2
    ry_new = ry + vy_new * dt + 0.5 * a_y * dt**2
    return jnp.array([x_h + rx_new, y_h + ry_new])


def compute_reconstruction_error(
    predicted_state_seq: jax.Array, position_seq: jax.Array, target_seq: jax.Array
):
    """Compute the reconstruction error between the predictions and the input sequence.

    Args:
        predicted_state_seq: State predictions in a single sequence.
        position_seq: Input positions in a single sequence.
        target_seq: Target positions in a single sequence.

    Returns:
        Reconstruction error (RMSE) for a sequence.
    """
    predictions = jax.vmap(reconstruct_next_position_from_state)(
        predicted_state_seq, position_seq
    )
    return jnp.sqrt(jnp.mean((predictions - target_seq) ** 2))


def compute_reconstruction_error_per_batch(
    predictions_batch: jax.Array, input_batch: jax.Array
):
    """Compute the reconstruction error between the predictions and the input sequence.

    Args:
        predictions_batch: State predictions in a single batch.
        input_batch: Input sequences in a single batch.

    Returns:
        Reconstruction error (RMSE) for a batch.
    """
    predictions_batch = predictions_batch[:, :-1, :]
    input_batch, target_batch = input_batch[:, :-1, :], input_batch[:, 1:, :]
    return jax.vmap(compute_reconstruction_error)(
        predictions_batch, input_batch, target_batch
    ).mean()
    

def compute_baseline_error(
    baseline_pred_seq, target_seq
):
    return jnp.sqrt(jnp.mean((baseline_pred_seq - target_seq) ** 2))


def compute_baseline_error_per_batch(
    input_batch: jax.Array
):
    baseline_pred_batch =  input_batch[:, :-1, :]
    target_batch = input_batch[:, 1:, :]
    return jax.vmap(compute_baseline_error)(
        baseline_pred_batch, target_batch
    ).mean()


def main(config: TransformerConfig):
    # Load the state predictions and turn them into observations
    if config.prediction_path:
        prediction_path = GEN_DATA_DIR / config.prediction_path
    else:
        prediction_path = GEN_DATA_DIR / "predictions"
    predictions, input_sequences, target_sequences = [], [], []
    for i in range(config.max_epochs):
        full_path = prediction_path / f"predictions_epoch_{i}.npy"
        print(f"Attempting to load file: {full_path}")
        prediction = np.load(full_path, allow_pickle=True)
        prediction = np.load(
            prediction_path / f"predictions_epoch_{i}.npy", allow_pickle=True
        )
        input_seq = np.load(
            prediction_path / f"inputs_epoch_{i}.npy", allow_pickle=True
        )
        target_seq = np.load(
            prediction_path / f"targets_epoch_{i}.npy", allow_pickle=True
        )
        predictions.append(prediction)
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    predictions = np.concatenate(predictions, axis=0)
    input_sequences = np.concatenate(input_sequences, axis=0)
    target_sequences = np.concatenate(target_sequences, axis=0)

    # nt_ablation_pos_heavy_errors = []
    # nt_ablation_vel_errors = []
    # nt_ablation_mass_errors = []
    # nt_reconstruction_errors = []  # noqa: FURB138
    # nt_baseline_errors = []  # noqa: FURB138
    # nt_ground_truth_errors = []  # noqa: FURB138
    ns_ablation_pos_heavy_errors = []
    ns_ablation_vel_errors = []
    ns_ablation_mass_errors = []
    for batch_pred, batch_inp, batch_target in tqdm.tqdm(
        zip(predictions, input_sequences, target_sequences), total=len(predictions)
    ):
        # nt_reconstruction_errors.append(
        #     compute_reconstruction_error_per_batch(
        #         jnp.array(batch_pred), jnp.array(batch_inp)
        #     )
        # )
        # nt_ground_truth_errors.append(
        #     compute_reconstruction_error_per_batch(
        #         jnp.array(batch_target), jnp.array(batch_inp)
        #     )
        # )
        # nt_baseline_errors.append(
        #     compute_baseline_error_per_batch(
        #         jnp.array(batch_inp)
        #     )
        # )
        # # Ablation 1: Keep ground truth/target_sequences except for position of heavy object
        # batch_target_ablation_pos_heavy = np.array(batch_target)
        # batch_target_ablation_pos_heavy[:, :, 2:4] = batch_pred[:, :, 2:4]
        # nt_ablation_pos_heavy_errors.append(
        #     compute_reconstruction_error_per_batch(
        #         jnp.array(batch_target_ablation_pos_heavy), jnp.array(batch_inp)
        #     )
        # )
        
        # # Ablation 2: Keep ground truth/target_sequences except for relative velocity
        # batch_target_ablation_vel = np.array(batch_target)
        # batch_target_ablation_vel[:, :, 4:6] = batch_pred[:, :, 4:6]
        # nt_ablation_vel_errors.append(
        #     compute_reconstruction_error_per_batch(
        #         jnp.array(batch_target_ablation_vel), jnp.array(batch_inp)
        #     )
        # )
        
        # # Ablation 3: Keep ground truth/target_sequences except for mass of heavier object
        # batch_target_ablation_mass = np.array(batch_target)
        # batch_target_ablation_mass[:, :, -1] = batch_pred[:, :, -1]
        # nt_ablation_mass_errors.append(
        #     compute_reconstruction_error_per_batch(
        #         jnp.array(batch_target_ablation_mass), jnp.array(batch_inp)
        #     )
        # )
        
        # # Ablation 1: check pos heavy state error
        # batch_pred_pos_heavy = jnp.array(batch_pred[:, :, 2:4])
        # batch_gt_pos_heavy = jnp.array(batch_target[:, :, 2:4])
        # ns_ablation_pos_heavy_errors.append(
        #     jax.vmap(compute_baseline_error)(
        #         batch_pred_pos_heavy, batch_gt_pos_heavy
        #     ).mean()
        # )
        
        # # Ablation 2: check velocity state error
        # batch_pred_vel = jnp.array(batch_pred[:, :, 4:6])
        # batch_gt_vel = jnp.array(batch_target[:, :, 4:6])
        # ns_ablation_vel_errors.append(
        #     jax.vmap(compute_baseline_error)(
        #         batch_pred_vel, batch_gt_vel
        #     ).mean()
        # )
        
        # # Ablation 3: check mass state error
        # batch_pred_mass = jnp.array(batch_pred[:, :, -1])
        # batch_gt_mass = jnp.array(batch_target[:, :, -1])
        # ns_ablation_mass_errors.append(
        #     jax.vmap(compute_baseline_error)(
        #         batch_pred_mass, batch_gt_mass
        #     ).mean()
        # )
        
        # Ablation: check mass state error
        batch_pred_mass = jnp.array(batch_pred)
        batch_gt_mass = jnp.array(batch_target)
        ns_ablation_mass_errors.append(
            jax.vmap(compute_baseline_error)(
                batch_pred_mass, batch_gt_mass
            ).mean()
        )
        

    # Save the reconstruction errors
    # np.save(
    #     prediction_path / "reconstruction_errors.npy", np.array(nt_reconstruction_errors)
    # )
    # np.save(prediction_path / "ground_truth_errors.npy", np.array(nt_ground_truth_errors))
    # np.save(prediction_path / "baseline_errors.npy", np.array(nt_baseline_errors))
    # np.save(
    #     prediction_path / "nt_ablation_pos_heavy_errors.npy", np.array(nt_ablation_pos_heavy_errors)
    # )
    # np.save(
    #     prediction_path / "nt_ablation_vel_errors.npy", np.array(nt_ablation_vel_errors)
    # )
    # np.save(
    #     prediction_path / "nt_ablation_mass_errors.npy", np.array(nt_ablation_mass_errors)
    # )
    # np.save(
    #     prediction_path / "ns_ablation_pos_heavy_errors.npy", np.array(ns_ablation_pos_heavy_errors)
    # )
    # np.save(
    #     prediction_path / "ns_ablation_vel_errors.npy", np.array(ns_ablation_vel_errors)
    # )
    np.save(
        prediction_path / "ns_ablation_mass_errors.npy", np.array(ns_ablation_mass_errors)
    )


if __name__ == "__main__":
    config = load_config("reconstruction_mass_config", logger)
    main(config)
