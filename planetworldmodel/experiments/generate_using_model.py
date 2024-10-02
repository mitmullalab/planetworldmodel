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


def evaluate_ntp_loss(model: TransformerRegressor, data: np.ndarray, device: torch.device) -> float:
    """Evaluate the Next-Token Prediction (NTP) loss on the given data.

    Args:
        model: The TransformerRegressor model.
        data: The validation data of shape (num_sequences, sequence_length, feature_dim).
        device: The device to perform computations on.

    Returns:
        The average NTP loss across all sequences.
    """
    model.eval()
    total_loss = 0.0
    num_sequences = len(data)

    with torch.no_grad():
        for sequence in tqdm.tqdm(data, desc="Evaluating NTP loss"):
            input_tensor = torch.tensor(sequence[:-1], dtype=torch.float32).unsqueeze(0).to(device)
            target_tensor = torch.tensor(sequence[1:], dtype=torch.float32).unsqueeze(0).to(device)

            output = model(input_tensor)
            loss = torch.sqrt(torch.nn.functional.mse_loss(output, target_tensor))
            total_loss += loss.item()

    average_loss = total_loss / num_sequences
    return average_loss


def evaluate_baseline_ntp_loss(data: np.ndarray) -> float:
    """Evaluate the baseline Next-Token Prediction (NTP) loss on the given data.
    The baseline predicts the next token to be the same as the current token.

    Args:
        data: The validation data of shape (num_sequences, sequence_length, feature_dim).

    Returns:
        The average baseline NTP loss across all sequences.
    """
    total_loss = 0.0
    num_sequences = len(data)

    for sequence in tqdm.tqdm(data, desc="Evaluating baseline NTP loss"):
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]
        
        # Predict next token to be the same as the current token
        predicted_sequence = input_sequence
        
        # Compute RMSE loss
        loss = np.sqrt(np.mean((predicted_sequence - target_sequence) ** 2))
        total_loss += loss

    average_loss = total_loss / num_sequences
    return average_loss


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
    # Random subset of 1,000 sequences
    idx = rng.choice(len(traj_light_full), 1_000, replace=False)
    traj_light_full = traj_light_full[idx]
    ntp_loss = evaluate_ntp_loss(model, traj_light_full, device)
    logger.info(f"NTP loss on validation set: {ntp_loss}")
    
    # Evaluate baseline NTP loss
    baseline_ntp_loss = evaluate_baseline_ntp_loss(traj_light_full)
    logger.info(f"Baseline NTP loss on validation set: {baseline_ntp_loss}")

    # Generate and store model predictions
    for i, (tl, th) in tqdm.tqdm(enumerate(zip(traj_light, traj_heavy)), total=args.num_sequences):
        np.save(GEN_DATA_DIR / f"traj_light_full_{i+1}.npy", tl)
        np.save(GEN_DATA_DIR / f"traj_heavy_{i+1}.npy", th)
        partial_traj_light = tl[: args.num_observed]
        # Generate the completed sequence
        completed_sequence = complete_sequence(
            model=model,
            input_sequence=partial_traj_light,
            desired_length=len(tl),
            device=device,
        )
        complete_tf_sequence = complete_sequence(
            model=model,
            input_sequence=partial_traj_light,
            desired_length=len(tl),
            device=device,
            teacher_forcing=True,
            ground_truth=tl,
        )
        np.save(GEN_DATA_DIR / f"traj_light_partial_{i+1}.npy", partial_traj_light)
        np.save(GEN_DATA_DIR / f"traj_light_predicted_{i+1}.npy", completed_sequence)
        np.save(
            GEN_DATA_DIR / f"traj_light_predicted_tf_{i+1}.npy", complete_tf_sequence
        )


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
