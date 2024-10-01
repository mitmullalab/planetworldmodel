import logging

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import ModelCheckpoint

from planetworldmodel import (
    SequenceDataModule,
    TransformerConfig,
    TwoBodyProblem,
    load_config,
    load_model,
    setup_wandb,
)
from planetworldmodel.setting import CKPT_DIR, DATA_DIR, GEN_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(config: TransformerConfig):
    torch.set_float32_matmul_precision("medium")
    devices = find_usable_cuda_devices()
    num_devices = len(devices)
    wandb_logger = setup_wandb(config)

    # Instantiate the data module
    batch_size = config.batch_size_per_device * num_devices
    data_dir = DATA_DIR / f"obs_var_{config.observation_variance:.5f}"
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
    best_checkpoint = ckpt_dir / "best.ckpt"
    resume_checkpoint = best_checkpoint if best_checkpoint.exists() else None
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Set up trainer and fit the model - generate state predictions
    trainer = Trainer(
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=devices,
        logger=wandb_logger,
        use_distributed_sampler=False,
        log_every_n_steps=20,
    )
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)


if __name__ == "__main__":
    config = load_config("reconstruction_mass_config", logger)
    main(config)
