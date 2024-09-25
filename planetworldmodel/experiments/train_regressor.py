import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import ModelCheckpoint

from planetworldmodel import TransformerConfig, load_config, setup_wandb
from planetworldmodel.setting import CKPT_DIR, DATA_DIR

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
        input_sequence = torch.tensor(sequence[:-1], dtype=torch.float32)
        target_sequence = torch.tensor(sequence[1:], dtype=torch.float32)
        return {
            "input_sequence": input_sequence,
            "target_sequence": target_sequence,
        }


class SequenceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        hidden_state=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if hidden_state:
            self.train_file = self.data_dir / "two_body_problem_hidden_state_train.npy"
            self.val_file = self.data_dir / "two_body_problem_hidden_state_val.npy"
        else:
            self.train_file = self.data_dir / "two_body_problem_train.npy"
            self.val_file = self.data_dir / "two_body_problem_val.npy"

    def setup(self, stage=None):
        self.train = SequenceDataset(self.train_file)
        self.val = SequenceDataset(self.val_file)

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
    """Implements the positional encoding as described in the original Transformer paper."""

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
    ):
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
        self.regressor = nn.Linear(dim_embedding, feature_dim)

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
        return loss

    def validation_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        target_sequence = batch["target_sequence"]
        predictions = self(input_sequence)
        loss = torch.sqrt(nn.MSELoss()(predictions, target_sequence))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def main(config: TransformerConfig):
    torch.set_float32_matmul_precision("medium")
    devices = find_usable_cuda_devices()
    num_devices = len(devices)
    wandb_logger = setup_wandb(config)

    # Set up checkpoint
    ckpt_dir = CKPT_DIR / config.name
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{step}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    last_checkpoint = ckpt_dir / "last.ckpt"
    resume_checkpoint = last_checkpoint if last_checkpoint.exists() else None

    # Instantiate the data module and the model
    batch_size = config.batch_size_per_device * num_devices
    data_dir = DATA_DIR / f"obs_var_{config.observation_variance:.5f}"
    data_module = SequenceDataModule(
        data_dir=data_dir, batch_size=batch_size, hidden_state=config.hidden_state
    )

    # Manually call setup to initialize the datasets
    data_module.setup()

    # Load a sample to get the feature dimension
    sample = data_module.train[0]
    feature_dim = sample["input_sequence"].shape[-1]
    model = TransformerRegressor(
        config.name,
        config.num_layers,
        config.num_heads,
        config.dim_embedding,
        feature_dim,
        config.learning_rate,
    )

    # Set up trainer and fit the model
    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=devices,
        logger=wandb_logger,
        use_distributed_sampler=False,
    )
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to the yaml config file")
    args = parser.parse_args()
    config = load_config(args.config_file, logger)
    main(config)
