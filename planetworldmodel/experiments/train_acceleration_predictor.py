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
from pytorch_lightning.callbacks import ModelCheckpoint

from planetworldmodel import TransformerConfig, load_config, setup_wandb
from planetworldmodel.setting import CKPT_DIR, DATA_DIR, GEN_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccelPredictionDataset(Dataset):
    def __init__(self, input_file_path: Path, target_file_path: Path):
        inputs = np.load(input_file_path)
        targets = np.atleast_3d(np.load(target_file_path))

        # Input state: lighter trajectory (num_data, seq_len, feature_dim)
        self.input_states = torch.tensor(inputs, dtype=torch.float32)

        # Target acceleration vector: (num_data, seq_len, feature_dim)
        self.target_states = torch.tensor(targets, dtype=torch.float32)
        print(f"test target: {targets.min():.2f} to {targets.max():.2f} with shape {targets.shape}")

    def __len__(self) -> int:
        return len(self.input_states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_sequence": self.input_states[idx],
            "target_sequence": self.target_states[idx],
        }


class SequenceAccelDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        prediction_target: Literal[
            "accel_fixed_pos", "accel_var_pos"
        ] = "accel_fixed_pos",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if prediction_target == "accel_fixed_pos":
            train_file = "obs_train_heavier_fixed.npy"
            train_target_file = "accel_train_heavier_fixed.npy"
            val_file = "obs_val_heavier_fixed.npy"
            val_target_file = "accel_val_heavier_fixed.npy"
        else:
            train_file = "obs_train.npy"
            train_target_file = "accel_train.npy"
            val_file = "obs_val.npy"
            val_target_file = "accel_val.npy"
        self.train_file = self.data_dir / train_file
        self.train_target_file = self.data_dir / train_target_file
        self.val_file = self.data_dir / val_file
        self.val_target_file = self.data_dir / val_target_file

    def setup(self, stage=None):
        self.train = AccelPredictionDataset(
            self.train_file, self.train_target_file
        )
        self.val = AccelPredictionDataset(
            self.val_file, self.val_target_file
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

    def test_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        target_sequence = batch["target_sequence"]
        predictions = self(input_sequence)
        loss = torch.sqrt(nn.MSELoss()(predictions, target_sequence))
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class BestModelPredictionSaver(ModelCheckpoint):
    def __init__(self, dirpath, filename, test_dataloader):
        super().__init__(dirpath=dirpath, filename=filename, save_top_k=1, monitor='val_loss', mode='min')
        self.test_dataloader = test_dataloader
        self.best_val_loss = float('inf')

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        current_val_loss = trainer.callback_metrics.get('val_loss')
        
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.save_predictions(trainer, pl_module)

    def save_predictions(self, trainer, pl_module):
        pl_module.eval()
        inputs = []
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in self.test_dataloader:
                input_sequence = batch["input_sequence"].to(pl_module.device)
                target_sequence = batch["target_sequence"]
                pred = pl_module(input_sequence)
                inputs.append(input_sequence.cpu().numpy())
                predictions.append(pred.cpu().numpy())
                targets.append(target_sequence.numpy())
        
        inputs = np.concatenate(inputs)
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        print(self.dirpath)
        print(inputs[-1,:10])
        print(targets[-1,:10])
        
                
        os.makedirs(self.dirpath, exist_ok=True)
        np.save(f"{self.dirpath}/best_inputs.npy", inputs)
        np.save(f"{self.dirpath}/best_predictions.npy", predictions)
        np.save(f"{self.dirpath}/best_targets.npy", targets)
        
        pl_module.train()


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
    num_devices = len(devices)
    wandb_logger = setup_wandb(config)

    # Instantiate the data module
    batch_size = config.batch_size_per_device * num_devices
    data_dir = DATA_DIR / f"accelerations"
    data_module = SequenceAccelDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        prediction_target=config.prediction_target,
    )
    
    # Create test dataloader
    if config.prediction_target == "accel_fixed_pos":
        test_file = "obs_test_heavier_fixed.npy"
        test_target_file = "accel_test_heavier_fixed.npy"
    else:
        test_file = "obs_test.npy"
        test_target_file = "accel_test.npy"
    test_data = AccelPredictionDataset(
        data_dir / test_file, data_dir / test_target_file,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_module.num_workers
    )

    # Manually call setup to initialize the datasets
    data_module.setup()

    # Load a sample to get the feature and output dimensions
    sample = data_module.train[0]
    feature_dim = sample["input_sequence"].shape[-1]
    output_dim = sample["target_sequence"].shape[-1]
    print(f"feature_dim: {feature_dim}, output_dim: {output_dim}")

    # Load the pretrained model
    model = load_model(config, feature_dim, output_dim)

    # Set up new checkpoint directory
    ckpt_name = config.new_ckpt_dir or config.name
    ckpt_dir = CKPT_DIR / ckpt_name
    best_checkpoint = ckpt_dir / "best.ckpt"
    resume_checkpoint = best_checkpoint if best_checkpoint.exists() else None
    
    
    # Modified checkpoint callback configuration
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch_{epoch:02d}",  # This will create checkpoints like "epoch_01.ckpt", "epoch_02.ckpt", etc.
        every_n_epochs=20,  # Save a checkpoint every epoch
        save_last=True,  # Still save the last checkpoint
    )
    
    best_model_saver = BestModelPredictionSaver(
        dirpath=ckpt_dir,
        filename="best",
        test_dataloader=test_dataloader
    )

    # Set up trainer and fit the model
    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback, best_model_saver],
        accelerator="gpu",
        precision="16-mixed",
        devices=devices,
        logger=wandb_logger,
        use_distributed_sampler=False,
        log_every_n_steps=20,
    )
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix_heavier_object_across_sequences",
        action="store_true",
    )
    parser.add_argument(
        "--use_sp_pretrained_model",
        action="store_true",
    )
    args = parser.parse_args()
    pt_name = "ntp"
    if args.use_sp_pretrained_model:
        pt_name = "sp"
    config_file = f"{pt_name}_pt_accel_transfer_var_pos_config"
    if args.fix_heavier_object_across_sequences:
        config_file = f"{pt_name}_pt_accel_transfer_fixed_pos_config" 
    config = load_config(config_file, logger)
    main(config)
