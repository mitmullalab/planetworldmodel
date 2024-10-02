import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import tqdm

from planetworldmodel.setting import CKPT_DIR, DATA_DIR, GEN_DATA_DIR
from planetworldmodel import TransformerConfig, TransformerRegressor, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model(
    config: TransformerConfig, checkpoint_path: Path, feature_dim: int
) -> TransformerRegressor:
    """Initialize the model and load weights from the checkpoint.

    Args:
        config: The TransformerConfig object.
        checkpoint_path: Path to the model checkpoint.

    Returns:
        The loaded TransformerRegressor model.
    """
    # Instantiate the model
    model = TransformerRegressor(
        model_name=config.name,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dim_embedding=config.dim_embedding,
        feature_dim=feature_dim,
        learning_rate=config.learning_rate,
    )

    # Load the checkpoint
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file {checkpoint_path} does not exist.")
        sys.exit(1)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Extract validation loss
    val_loss = checkpoint.get("val_loss", None)
    if val_loss is None:
        logger.warning("Validation loss not found in the checkpoint.")
    else:
        logger.info(f"Validation loss from checkpoint: {val_loss}")

    return model


def compute_multi_step_error(model, sequence, device, steps):
    """Compute multi-step prediction error.

    Args:
        model: The TransformerRegressor model.
        sequence: A single sequence of shape (sequence_length, feature_dim).
        device: The device to perform computations on.
        steps: Number of steps to predict ahead.

    Returns:
        The average multi-step prediction error for the sequence.
    """
    total_error = 0.0
    num_predictions = len(sequence) - steps

    for i in range(num_predictions):
        input_tensor = torch.tensor(sequence[i:i+1], dtype=torch.float32).unsqueeze(0).to(device)
        target = sequence[i+steps]

        with torch.no_grad():
            for _ in range(steps):
                output = model(input_tensor)
                next_step = output[0, -1, :].cpu().numpy()
                input_tensor = torch.cat([input_tensor, torch.tensor(next_step).unsqueeze(0).unsqueeze(0).to(device)], dim=1)

        prediction = input_tensor[0, -1, :].cpu().numpy()
        error = np.sqrt(np.mean((prediction - target) ** 2))
        total_error += error

    return total_error / num_predictions


def evaluate_model_performance(model, data, device, steps_ahead=(1, 2, 5, 10)):
    """Evaluate the model's performance using multi-step predictions for specified steps ahead.

    Args:
        model: The TransformerRegressor model.
        data: The validation data of shape (num_sequences, sequence_length, feature_dim).
        device: The device to perform computations on.
        steps_ahead: Tuple of step sizes to evaluate.

    Returns:
        A dictionary containing the computed metrics.
    """
    model.eval()
    metrics = {f"{step}_step_error": [] for step in steps_ahead}
    num_sequences, seq_length, feature_dim = data.shape

    with torch.no_grad():
        for sequence in tqdm.tqdm(data, desc="Evaluating model performance"):
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            
            for t in range(seq_length - max(steps_ahead)):
                input_tensor = sequence_tensor[:, :t+1, :]
                
                # Generate predictions for all steps ahead
                predictions = []
                for _ in range(max(steps_ahead)):
                    output = model(input_tensor)
                    next_pred = output[:, -1:, :]
                    predictions.append(next_pred)
                    input_tensor = torch.cat([input_tensor, next_pred], dim=1)
                
                predictions = torch.cat(predictions, dim=1)
                
                # Compute errors for specified steps
                for step in steps_ahead:
                    pred = predictions[:, step-1, :]
                    target = sequence_tensor[:, t+step, :]
                    error = torch.sqrt(torch.mean((pred - target) ** 2))
                    metrics[f"{step}_step_error"].append(error.item())

    # Compute average metrics
    for key in metrics:
        metrics[key] = np.mean(metrics[key])

    return metrics


def evaluate_baseline_performance(data, steps_ahead=(1, 2, 5, 10)):
    """Evaluate the baseline model's performance using various metrics for specified steps ahead.

    Args:
        data: The validation data of shape (num_sequences, sequence_length, feature_dim).
        steps_ahead: Tuple of step sizes to evaluate.

    Returns:
        A dictionary containing the computed metrics for the baseline.
    """
    metrics = {f"{step}_step_error": 0.0 for step in steps_ahead}
    num_sequences, seq_length, feature_dim = data.shape

    for sequence in tqdm.tqdm(data, desc="Evaluating baseline performance"):
        for step in steps_ahead:
            predictions = sequence[:-step]
            targets = sequence[step:]
            error = np.sqrt(np.mean((predictions - targets) ** 2))
            metrics[f"{step}_step_error"] += error

    # Average the metrics
    for key in metrics:
        metrics[key] /= num_sequences

    return metrics


def complete_sequence(
    model: TransformerRegressor,
    input_sequence: np.ndarray,
    desired_length: int = 1000,
    device: torch.device = torch.device("cpu"),
    teacher_forcing: bool = False,
    ground_truth: np.ndarray | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Autoregressively generate a sequence to reach the desired length.

    Args:
        model: The trained TransformerRegressor model.
        input_sequence: Initial sequence of shape (x, 4) where x <= 1000.
        desired_length: The target sequence length.
        device: The device to perform computations on.
        teacher_forcing: If True, use ground truth for input at each step.
        ground_truth: The full ground truth sequence, required if teacher_forcing is True.


    Returns:
        The completed sequence of shape (desired_length, 4).
    """
    sequence = input_sequence.copy()
    logger.info(f"Initial sequence length: {sequence.shape[0]}")

    if teacher_forcing and ground_truth is None:
        raise ValueError("Ground truth must be provided when teacher_forcing is True.")

    with torch.no_grad():
        while len(sequence) < desired_length:
            # Prepare input tensor
            if teacher_forcing:
                assert ground_truth is not None
                input_tensor = (
                    torch.tensor(ground_truth[: len(sequence)], dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
            else:
                input_tensor = (
                    torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
                )  # (1, seq_len, 4)

            # Forward pass
            output = model(input_tensor)  # (1, seq_len, 4)

            # Take the last prediction
            next_step = output[0, -1, :].cpu().numpy()

            # Append the next step to the sequence
            sequence = np.vstack([sequence, next_step])

            if verbose and (
                len(sequence) % 100 == 0 or len(sequence) == desired_length
            ):
                logger.info(f"Generated {len(sequence)} / {desired_length} steps")

    logger.info(f"Completed sequence generation. Final length: {sequence.shape[0]}")
    return sequence


def main(args):
    # Load configuration
    config = load_config(args.config_file, logger)
    logger.info("Configuration loaded.")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    rng = np.random.default_rng(args.seed)
    data_dir = DATA_DIR / "two_body_problem"
    traj_light_full = np.load(data_dir / "obs_test_heavier_fixed.npy")
    state = np.load(data_dir / "state_test_heavier_fixed.npy")
    traj_heavy = state[:, :, 2:4]
    idx = rng.choice(len(traj_light_full), args.num_sequences, replace=False)
    traj_light = traj_light_full[idx]
    traj_heavy = traj_heavy[idx]

    # Initialize and load the model
    feature_dim = traj_light.shape[-1]
    checkpoint_path = (
        CKPT_DIR / config.name / "best.ckpt"
        if not args.checkpoint
        else Path(args.checkpoint)
    )
    model = setup_model(config, checkpoint_path, feature_dim).to(device)
    logger.info("Model loaded and ready for generation.")
    
    # Evaluate NTP loss on the validation set
    # Random subset of 100 sequences
    idx = rng.choice(len(traj_light_full), 100, replace=False)
    traj_light_full = traj_light_full[idx]
    
    # Evaluate baseline performance
    baseline_metrics = evaluate_baseline_performance(traj_light_full)
    logger.info("Baseline performance:")
    for key, value in baseline_metrics.items():
        logger.info(f"{key}: {value}")
    # Save
    np.save(GEN_DATA_DIR / "baseline_metrics.npy", baseline_metrics)
    with open(GEN_DATA_DIR / "baseline_metrics.txt", "w") as f:
        for key, value in baseline_metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Evaluate model performance
    model_metrics = evaluate_model_performance(model, traj_light_full, device)
    logger.info("Model performance:")
    for key, value in model_metrics.items():
        logger.info(f"{key}: {value}")
    # Save
    np.save(GEN_DATA_DIR / "model_metrics.npy", model_metrics)
    with open(GEN_DATA_DIR / "model_metrics.txt", "w") as f:
        for key, value in model_metrics.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a complete sequence using a trained TransformerRegressor model."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Name of the YAML config file without the extension.",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num_observed",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint. If not provided, it uses the last.ckpt in the config's checkpoint directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    main(args)
