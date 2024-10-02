import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import ModelCheckpoint

from planetworldmodel import TransformerConfig, load_config, setup_wandb
from planetworldmodel.setting import CKPT_DIR, DATA_DIR, GEN_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    def __init__(self, file_path: Path):
        self.data = np.load(file_path)
        self.seq_len = self.data.shape[1]
        self.feature_dim = self.data.shape[2]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sequence = self.data[idx]
        input_sequence = torch.tensor(sequence, dtype=torch.float32)
        return {
            "input_sequence": input_sequence,
        }
        

class SequenceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        split: Literal["train", "val"] = "train",
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if split == "train":
            train_file = "obs_train_heavier_fixed.npy"
        else:
            train_file = "obs_val_heavier_fixed.npy"
        self.train_file = self.data_dir / train_file

    def setup(self, stage=None):
        self.train = SequenceDataset(self.train_file)

    def predict_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransformerRegressor(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_layers: int,
        num_heads: int,
        dim_embedding: int,
        feature_dim: int,
        learning_rate: float,
        max_seq_length: int = 5000,
        output_dim: int | None = None,
        store_predictions: bool = False,
        prediction_path: Path | str | None = "predictions",
    ):
        if output_dim is None:
            output_dim = feature_dim
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.embedding = nn.Linear(feature_dim, dim_embedding)
        self.positional_encoding = PositionalEncoding(
            dim_embedding, max_len=max_seq_length
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads),
            num_layers=num_layers,
        )
        self.regressor = nn.Linear(dim_embedding, output_dim)
        self.store_predictions = store_predictions
        self.prediction_path = prediction_path
        self.inputs: list = []
        self.predictions: list = []
        self.targets: list = []
        self.output_dim = output_dim

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, dim_embedding)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, dim_embedding) # noqa: FURB184

        # Generate causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)

        # memory is None because we are not using encoder
        x = self.transformer(x, mask=mask)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, dim_embedding)  # noqa: FURB184
        x = self.regressor(x)
        return x[..., :self.output_dim]
    
    def predict_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        predictions = self(input_sequence)
        return predictions

    def configure_optimizers(self):
        return None  # No need to define optimizers for evaluation

        
def load_model(
    config: TransformerConfig,
    feature_dim: int,
    ckpt_filename: str,
    num_data_points: int,
) -> TransformerRegressor:
    """Load the model with the pretrained weights, if they exist.
    Otherwise, initialize the model with random weights.

    Args:
        config: The config object.
        feature_dim: The dimension of the input features.
        output_dim: The dimension of the output features.

    Returns:
        The TransformerRegressor model object.
    """
    # Initialize the model
    model = TransformerRegressor(
        config.name,
        config.num_layers,
        config.num_heads,
        config.dim_embedding,
        feature_dim,
        config.learning_rate,
        store_predictions=config.store_predictions,
        prediction_path=config.prediction_path,
        output_dim=1,
    )
    ckpt_name = config.name
    ckpt_path = CKPT_DIR / ckpt_name / f"data_points_{num_data_points}" / f"{ckpt_filename}.ckpt"
    if ckpt_path and ckpt_path.exists():
        try:
            checkpoint = torch.load(ckpt_path, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            logger.info(f"Loaded pretrained weights from {ckpt_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Initializing model with random weights")
    else:
        logger.info(
            "No pretrained checkpoint provided or file not found. "
            "Initializing model with random weights"
        )
    return model


def load_model_and_generate_predictions(
    pretrained: Literal["sp", "ntp", "sp_direct"],
    split: Literal["train", "val"],
    model_num: int,
    checkpoint_epoch_num: int,
    num_data_points: int,
):
    devices = [0]  # Use a single GPU
    num_devices = len(devices)
    config_file = f"ntp_pt_noise_finetune{model_num}_config"
    if pretrained == "sp":
        config_file = f"sp_pt_noise_finetune{model_num}_config"
    elif pretrained == "sp_direct":
        config_file = f"sp_pt_direct_noise_finetune_config"
    config = load_config(config_file, logger)

    # Instantiate the data module
    batch_size = config.batch_size_per_device * num_devices
    data_dir = DATA_DIR / f"two_body_problem"
    data_module = SequenceDataModule(
        data_dir=data_dir,
        split=split,
        batch_size=batch_size,
    )

    # Manually call setup to initialize the datasets
    data_module.setup()

    # Load a sample to get the feature and output dimensions

    sample = data_module.train[0]
    feature_dim = sample["input_sequence"].shape[-1]
    
    ckpt_file_name = f"{pretrained}_pt_noise_finetune{model_num}-1mil_epoch_{checkpoint_epoch_num}"
    
    model = load_model(config, feature_dim, ckpt_file_name, num_data_points)
    
    # Set up trainer for prediction
    trainer = Trainer(
        accelerator="gpu",
        precision="16-mixed",
        devices=devices,
    )
    
    # Generate predictions
    batched_predictions = trainer.predict(model, data_module)

    # Concatenate all batches into a single numpy array
    all_predictions = np.concatenate([
        batch.cpu().numpy() for batch in batched_predictions
    ], axis=0)

    
    return all_predictions, data_dir


def main(args):
    torch.set_float32_matmul_precision("medium")
    
    # Generate predictions
    predictions = []
    for model_num in range(1, args.num_models + 1):
        prediction, data_dir = load_model_and_generate_predictions(
            args.pretrained, args.split, model_num, args.checkpoint_epoch_num,
            args.num_data_points
        )
        print(prediction.shape)
        predictions.append(prediction)
        
    predictions = np.concatenate(predictions, axis=-1)

    # Save predictions
    np.save(
        data_dir / f"{args.pretrained}_noise_trained_model_predictions_{args.split}_epoch_{args.checkpoint_epoch_num}_data_points_{args.num_data_points}.npy",
        predictions
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        type=str,
        choices=["sp", "ntp", "sp_direct"],
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
    )
    parser.add_argument(
        "--checkpoint_epoch_num",
        type=int,
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num_data_points",
        type=int,
    )
    args = parser.parse_args()
    main(args)