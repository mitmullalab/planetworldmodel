import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import ModelCheckpoint

from planetworldmodel import TransformerConfig, load_config, setup_wandb
from planetworldmodel.setting import CKPT_DIR, DATA_DIR, GEN_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NextTokenPredictionDataset(Dataset):
    def __init__(self, file_path: Path):
        self.data = np.load(file_path)
        self.seq_len = self.data.shape[1]
        self.feature_dim = self.data.shape[2]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sequence = self.data[idx]
        input_sequence = torch.tensor(sequence[:-1], dtype=torch.float32)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.float32)
        return {
            "input_sequence": input_sequence,
            "target_sequence": target_sequence,
        }


class StatePredictionDataset(Dataset):
    def __init__(self, file_path: Path):
        states = np.load(file_path)

        # Input state: lighter trajectory (num_data, seq_len, feature_dim)
        self.input_states = torch.tensor(states[:, :, :2], dtype=torch.float32)

        # Target state: (num_data, seq_len, 3*feature_dim+2)
        # lighter trajectory, heavier trajectory, relative velocity, log(m1), log(m2)
        self.target_states = torch.tensor(states[:, :, :-2], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.input_states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_sequence": self.input_states[idx],
            "target_sequence": self.target_states[idx],
        }
        

class MassPredictionDataset(Dataset):
    def __init__(self, file_path: Path):
        states = np.load(file_path)

        # Input state: lighter trajectory (num_data, seq_len, feature_dim)
        self.input_states = torch.tensor(states[:, :, :2], dtype=torch.float32)

        # Target state: (num_data, seq_len, 1)
        self.target_states = torch.tensor(states[:, :, -3:-2], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.input_states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_sequence": self.input_states[idx],
            "target_sequence": self.target_states[idx],
        }
        

class FunctionOfStatePredictionDataset(Dataset):
    def __init__(self, file_path: Path):
        states = np.load(file_path)

        # Input state: lighter trajectory (num_data, seq_len, feature_dim)
        self.input_states = torch.tensor(states[:, :, :2], dtype=torch.float32)

        # Target state: (num_data, seq_len, 2)
        # log(energy), log(angular_momentum)
        self.target_states = torch.tensor(states[:, :, -2:], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.input_states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_sequence": self.input_states[idx],
            "target_sequence": self.target_states[idx],
        }


class SequenceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        prediction_target: Literal[
            "next_obs", "state", "function_of_state", "mass"
        ] = "next_obs",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if prediction_target == "next_obs":
            train_file = "obs_train_heavier_fixed.npy"
            val_file = "obs_val_heavier_fixed.npy"
        else:
            train_file = "standardized_state_train_heavier_fixed.npy"
            val_file = "standardized_state_val_heavier_fixed.npy"
        self.train_file = self.data_dir / train_file
        self.val_file = self.data_dir / val_file
        self.prediction_target = prediction_target

    def setup(self, stage=None):
        if self.prediction_target == "next_obs":
            self.train = NextTokenPredictionDataset(self.train_file)
            self.val = NextTokenPredictionDataset(self.val_file)
        elif self.prediction_target == "state":
            self.train = StatePredictionDataset(self.train_file)
            self.val = StatePredictionDataset(self.val_file)
        elif self.prediction_target == "mass":
            self.train = MassPredictionDataset(self.train_file)
            self.val = MassPredictionDataset(self.val_file)
        else:
            self.train = FunctionOfStatePredictionDataset(self.train_file)
            self.val = FunctionOfStatePredictionDataset(self.val_file)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
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

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, dim_embedding)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, dim_embedding) # noqa: FURB184

        # Generate causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)

        # memory is None because we are not using encoder
        x = self.transformer(x, mask=mask)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, dim_embedding)  # noqa: FURB184
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        target_sequence = batch["target_sequence"]
        predictions = self(input_sequence)
        loss = torch.sqrt(nn.MSELoss()(predictions, target_sequence))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        if self.store_predictions:
            self.predictions.append(predictions.detach().cpu().numpy())
            self.inputs.append(input_sequence.detach().cpu().numpy())
            self.targets.append(target_sequence.detach().cpu().numpy())
        return loss

    def on_train_epoch_end(self):
        if not self.store_predictions:
            return

        epoch_predictions = np.array(self.predictions, dtype=object)
        epoch_inputs = np.array(self.inputs, dtype=object)
        epoch_targets = np.array(self.targets, dtype=object)

        # Save predictions and inputs
        if self.prediction_path:
            prediction_path = GEN_DATA_DIR / self.prediction_path
        else:
            prediction_path = GEN_DATA_DIR / "predictions"
        prediction_path.mkdir(exist_ok=True, parents=True)
        np.save(
            prediction_path / f"predictions_epoch_{self.current_epoch}.npy",
            epoch_predictions,
        )
        np.save(
            prediction_path / f"inputs_epoch_{self.current_epoch}.npy", epoch_inputs
        )
        np.save(
            prediction_path / f"targets_epoch_{self.current_epoch}.npy", epoch_targets
        )

        # Clear for next epoch
        self.predictions = []
        self.inputs = []
        self.targets = []

    def validation_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        target_sequence = batch["target_sequence"]
        predictions = self(input_sequence)
        loss = torch.sqrt(nn.MSELoss()(predictions, target_sequence))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        print(f"Initial learning_rate: {self.learning_rate}")
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }


def load_model(
    config: TransformerConfig,
    feature_dim: int,
    output_dim: int,
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
    pretrained_output_dim = config.pretrained_output_dim or output_dim

    # Initialize the model
    model = TransformerRegressor(
        config.name,
        config.num_layers,
        config.num_heads,
        config.dim_embedding,
        feature_dim,
        config.learning_rate,
        output_dim=pretrained_output_dim,
        store_predictions=config.store_predictions,
        prediction_path=config.prediction_path,
    )

    # Load the pretrained weights, if they exist
    ckpt_name = config.pretrained_ckpt_dir or config.name
    ckpt_path = CKPT_DIR / ckpt_name / "best.ckpt"
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
    # Replace the final regressor layer with a new one matching the new output_dim
    # if output_dim != pretrained_output_dim:
    model.regressor = nn.Linear(config.dim_embedding, output_dim)
    return model


def main(config: TransformerConfig):
    torch.set_float32_matmul_precision("medium")
    devices = find_usable_cuda_devices()
    print(f"Devices: {devices}")
    num_devices = len(devices)
    wandb_logger = setup_wandb(config)

    # Instantiate the data module
    batch_size = config.batch_size_per_device * num_devices
    # data_dir = DATA_DIR / f"obs_var_{config.observation_variance:.5f}"
    data_dir = DATA_DIR / f"two_body_problem"
    data_module = SequenceDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        prediction_target=config.prediction_target,
    )

    # Manually call setup to initialize the datasets
    data_module.setup()

    # Load a sample to get the feature and output dimensions
    sample = data_module.train[0]
    feature_dim = sample["input_sequence"].shape[-1]
    output_dim = sample["target_sequence"].shape[-1]

    # Load the pretrained model
    model = load_model(config, feature_dim, output_dim)

    # Set up new checkpoint directory
    ckpt_name = config.new_ckpt_dir or config.name
    ckpt_dir = CKPT_DIR / ckpt_name
    # best_checkpoint = ckpt_dir / "best.ckpt"
    # resume_checkpoint = best_checkpoint if best_checkpoint.exists() else None
    last_checkpoint = ckpt_dir / "last.ckpt"
    resume_checkpoint = last_checkpoint if last_checkpoint.exists() else None
    
    # Modified checkpoint callback configuration
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch_{epoch:02d}",  # This will create checkpoints like "epoch_01.ckpt", "epoch_02.ckpt", etc.
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=20,  # Save a checkpoint every epoch
        save_last=True,  # Still save the last checkpoint
    )

    # Additional checkpoint callback for saving the best model
    best_model_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # Set up trainer and fit the model
    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, best_model_callback],  # Add both callbacks
        accelerator="gpu",
        precision="16-mixed",
        devices=devices,
        logger=wandb_logger,
        use_distributed_sampler=True,
        log_every_n_steps=20,
    )
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to the yaml config file")
    args = parser.parse_args()
    config = load_config(args.config_file, logger)
    main(config)
