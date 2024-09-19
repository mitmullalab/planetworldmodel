import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch

from planetworldmodel.setting import CKPT_DIR, GEN_DATA_DIR
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
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
    )
    model.eval()

    return model


def generate_sequence(
    model: TransformerRegressor,
    input_sequence: np.ndarray,
    desired_length: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Autoregressively generate a sequence to reach the desired length.

    Args:
        model: The trained TransformerRegressor model.
        input_sequence: Initial sequence of shape (x, 4) where x <= 1000.
        desired_length: The target sequence length.
        device: The device to perform computations on.

    Returns:
        The completed sequence of shape (desired_length, 4).
    """
    sequence = input_sequence.copy()
    logger.info(f"Initial sequence length: {sequence.shape[0]}")

    with torch.no_grad():
        while len(sequence) < desired_length:
            # Prepare input tensor
            input_tensor = (
                torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            )  # (1, seq_len, 4)

            # Forward pass
            output = model(input_tensor)  # (1, seq_len, 4)

            # Take the last prediction
            next_step = output[0, -1, :].cpu().numpy()

            # Append the next step to the sequence
            sequence = np.vstack([sequence, next_step])

            if len(sequence) % 100 == 0 or len(sequence) == desired_length:
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

    # Load input sequence
    input_sequence = np.load(GEN_DATA_DIR / args.input_sequence)
    if input_sequence.ndim != 2 or input_sequence.shape[1] != 4:
        logger.error("Input sequence must have shape (x, 4).")
        sys.exit(1)
    if input_sequence.shape[0] > 1000:
        logger.error("Input sequence length exceeds 1000.")
        sys.exit(1)
    logger.info(f"Input sequence loaded with shape {input_sequence.shape}.")
    feature_dim = input_sequence.shape[1]

    # Initialize and load the model
    checkpoint_path = (
        CKPT_DIR / config.name / "last.ckpt"
        if not args.checkpoint
        else Path(args.checkpoint)
    )
    model = setup_model(config, checkpoint_path, feature_dim).to(device)
    logger.info("Model loaded and ready for generation.")

    # Generate the completed sequence
    completed_sequence = generate_sequence(
        model=model, input_sequence=input_sequence, desired_length=1000, device=device
    )

    # Save the completed sequence
    output_path = GEN_DATA_DIR / args.output_file
    np.save(output_path, completed_sequence)
    logger.info(f"Completed sequence saved to {output_path}.")


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
        "--input_sequence",
        type=str,
        required=True,
        help="Path to the input numpy file containing the partial sequence (x, 4).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the completed sequence as a numpy file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model checkpoint. If not provided, it uses the last.ckpt in the config's checkpoint directory.",
    )

    args = parser.parse_args()
    main(args)
