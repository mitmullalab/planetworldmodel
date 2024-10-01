import argparse
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from planetworldmodel import TransformerConfig, load_config, setup_wandb
from planetworldmodel.setting import CKPT_DIR, DATA_DIR, GEN_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoisePredictionDataset(Dataset):
    def __init__(
        self, 
        input_file_path: Path, 
        target_file_path: Path, 
        index_file_path: Path,
        num_data_points: int = 500,
        seed: int = 0,
    ):
        inputs = np.load(input_file_path)
        targets = np.load(target_file_path)
        indx = np.load(index_file_path)
        
        # # Extract each element corresponding to the indx
        # inputs = inputs[row_indices, indx]
        # targets = targets[row_indices, indx]
        
        # Extract random num_datapoints
        rng = np.random.default_rng(seed)
        rand_ints = rng.choice(len(inputs), num_data_points, replace=False)
        
        # Input state: lighter trajectory (num_data, seq_len, feature_dim)
        self.input_states = torch.tensor(inputs[rand_ints], dtype=torch.float32)

        # Target noise vector: (num_data, seq_len, 1)
        self.target_states = torch.tensor(targets[rand_ints], dtype=torch.float32)
        
        # Index of the target state in the input state
        self.index = torch.tensor(indx[rand_ints], dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.input_states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_sequence": self.input_states[idx],
            "target_sequence": self.target_states[idx],
            "index": self.index[idx],
        }


class SequenceNoiseDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        train_file = "obs_train_heavier_fixed.npy"
        train_target_file = "noise_train_heavier_fixed.npy"
        train_idx_file = "indx_train_heavier_fixed.npy"
        val_file = "obs_val_heavier_fixed.npy"
        val_target_file = "noise_val_heavier_fixed.npy"
        val_idx_file = "indx_val_heavier_fixed.npy"
        self.train_file = self.data_dir / train_file
        self.train_target_file = self.data_dir / train_target_file
        self.train_idx_file = self.data_dir / train_idx_file
        self.val_file = self.data_dir / val_file
        self.val_target_file = self.data_dir / val_target_file
        self.val_idx_file = self.data_dir / val_idx_file
        self.seed = seed

    def setup(self, stage=None):
        self.train = NoisePredictionDataset(
            self.train_file, self.train_target_file, self.train_idx_file, seed=self.seed
        )
        self.val = NoisePredictionDataset(
            self.val_file, self.val_target_file, self.val_idx_file, seed=self.seed
        )

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

    def training_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        target_sequence = batch["target_sequence"]
        indx = batch["index"]
        # print(f"input_seq: {input_sequence.shape}, target_seq: {target_sequence.shape}, indx: {indx.shape}")
        predictions = self(input_sequence)
        extracted_pred = predictions[torch.arange(predictions.size(0)), indx]
        extracted_target = target_sequence[torch.arange(target_sequence.size(0)), indx]
        # print(f"new_pred: {extracted_pred.shape}, new_target: {extracted_target.shape}")
        loss = torch.sqrt(nn.MSELoss()(extracted_pred, extracted_target))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        if self.store_predictions:
            self.predictions.append(predictions.detach().cpu().numpy())
            self.inputs.append(input_sequence.detach().cpu().numpy())
            self.targets.append(target_sequence.detach().cpu().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        target_sequence = batch["target_sequence"]
        indx = batch["index"]
        predictions = self(input_sequence)
        extracted_pred = predictions[torch.arange(predictions.size(0)), indx]
        extracted_target = target_sequence[torch.arange(target_sequence.size(0)), indx]
        loss = torch.sqrt(nn.MSELoss()(extracted_pred, extracted_target))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_model(
    config: TransformerConfig,
    feature_dim: int,
    output_dim: int,
    swap_final_layer: bool = True,
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
    if swap_final_layer:
        model.regressor = nn.Linear(config.dim_embedding, output_dim)
    return model


class CustomEpochCheckpoint(Callback):
    def __init__(self, dirpath, filename_prefix, save_epochs):
        super().__init__()
        self.dirpath = dirpath
        self.filename_prefix = filename_prefix
        self.save_epochs = save_epochs
        
    def on_fit_start(self, trainer, pl_module):
        # Save checkpoint before training starts (epoch 0)
        self._save_checkpoint(trainer, 0)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch + 1 in self.save_epochs:  # +1 because epochs are 0-indexed
            filename = f"{self.filename_prefix}_epoch_{epoch+1}.ckpt"
            ckpt_path = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            print(f"Saved checkpoint for epoch {epoch+1} at {ckpt_path}")
            
    def _save_checkpoint(self, trainer, epoch):
        filename = f"{self.filename_prefix}_epoch_{epoch}.ckpt"
        ckpt_path = os.path.join(self.dirpath, filename)
        trainer.save_checkpoint(ckpt_path)
        print(f"Saved checkpoint for epoch {epoch} at {ckpt_path}")


class SaveFinalCheckpoint(Callback):
    def __init__(self, dirpath, filename):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint(os.path.join(self.dirpath, f"{self.filename}.ckpt"))


def main(config: TransformerConfig, pretrained: str, num_data_points: int):
    torch.set_float32_matmul_precision("medium")
    devices = find_usable_cuda_devices()
    num_devices = len(devices)
    wandb_logger = setup_wandb(config)

    # Instantiate the data module
    batch_size = config.batch_size_per_device * num_devices
    data_dir = DATA_DIR / f"white_noise_data"
    data_module = SequenceNoiseDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=config.seed,
    )

    # Manually call setup to initialize the datasets
    data_module.setup()

    # Load a sample to get the feature and output dimensions
    sample = data_module.train[0]
    feature_dim = sample["input_sequence"].shape[-1]
    output_dim = sample["target_sequence"].shape[-1]
    # print(f"feature_dim: {feature_dim}, output_dim: {output_dim}")

    # Load the pretrained model
    swap_final_layer = True
    if pretrained == "sp_direct":
        swap_final_layer = False
    model = load_model(config, feature_dim, output_dim, swap_final_layer)

    # Set up new checkpoint directory
    ckpt_name = config.new_ckpt_dir or config.name
    ckpt_dir = CKPT_DIR / ckpt_name
    # Create custom callback to save final checkpoint
    # save_final_checkpoint = SaveFinalCheckpoint(dirpath=ckpt_dir, filename="last")
    custom_epoch_checkpoint = CustomEpochCheckpoint(
        dirpath=ckpt_dir,
        filename_prefix=config.name,
        save_epochs=[0, 1, 10, 50, 100]
    )

    # Set up trainer and fit the model
    trainer = Trainer(
        max_epochs=config.max_epochs,
        # callbacks=[save_final_checkpoint],
        callbacks=[custom_epoch_checkpoint],
        accelerator="gpu",
        precision="16-mixed",
        devices=devices,
        logger=wandb_logger,
        use_distributed_sampler=False,
        log_every_n_steps=1_000,
        limit_val_batches=0.0
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        type=str,
        choices=["sp", "ntp", "sp_direct"],
    )
    parser.add_argument(
        "--model_num",
        type=int,
        choices=[1,2,3,4,5,6,7,8,9,10],
    )
    parser.add_argument(
        "--num_data_points",
        type=int,
    )
    args = parser.parse_args()
    config_file = f"ntp_pt_noise_finetune{args.model_num}_config"
    if args.pretrained == "sp":
        config_file = f"sp_pt_noise_finetune{args.model_num}_config"
    elif args.pretrained == "sp_direct":
        config_file = f"sp_pt_direct_noise_finetune_config"
    config = load_config(config_file, logger)
    main(config, args.pretrained, args.num_data_points)
