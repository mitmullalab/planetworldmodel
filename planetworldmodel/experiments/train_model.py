import argparse
from functools import partial
import logging
from pathlib import Path
import pickle
import sys
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from pydantic import BaseModel, Field
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerConfig(BaseModel):
    name: str
    num_layers: int
    num_heads: int
    dim_embedding: int
    batch_size_per_device: int
    learning_rate: float
    max_epochs: int
    use_wandb: bool
    wandb_project: str = Field(
        "", description="Wandb project name. If use_wandb is False, this is ignored."
    )
    wandb_entity: str = Field(
        "", description="Wandb entity name. If use_wandb is False, this is ignored."
    )


class Tokenizer:
    def __init__(self, pad_token_id: int = 0, num_states: int = 10):
        self.pad_token_id = pad_token_id
        word_to_id = {"L": 1, "O": 2, "R": 3, "<pad>": self.pad_token_id}
        for state, next_id in zip(range(num_states), range(4, num_states + 4)):
            word_to_id[str(state)] = next_id
        self.word_to_id = word_to_id
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

    def encode(self, sentence: str) -> list[int]:
        return [
            self.word_to_id.get(word, self.word_to_id["<pad>"])
            for word in sentence.split()
        ]

    def decode(self, token_ids: torch.Tensor | np.ndarray) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        if token_ids.ndim == 0:
            token_ids = np.array([token_ids])
        return " ".join(
            self.id_to_word[id] for id in token_ids if id != self.pad_token_id
        )


class TextDataset(Dataset):
    def __init__(self, file_path: Path):
        with file_path.open("rb") as f:
            self.tokenized_sentences = pickle.load(f)

    def __len__(self) -> int:
        return len(self.tokenized_sentences)

    def __getitem__(self, idx: int) -> dict:
        token_ids = self.tokenized_sentences[idx]
        attention_mask = [1] * len(token_ids)  # Why?
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": token_ids,  # Why is this the same as input_ids?
        }


class DataModule(LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 4, num_devices: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.num_shards = num_devices
        self.data_dir = data_dir
        self.tokenizer = Tokenizer()
        self.collate_fn = partial(collate_fn, pad_token_id=self.tokenizer.pad_token_id)

    def prepare_data(self):
        """If tokenized data does not exist, prepare the data."""
        tokenized_train_path = self.data_dir / "train_tokenized.pkl"
        tokenized_val_path = self.data_dir / "val_tokenized.pkl"
        if tokenized_train_path.exists() and tokenized_val_path.exists():
            return

        logger.info("Loading datasets...")
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{self.data_dir}/train.txt", "r") as f:
            train_sequences = f.read().splitlines()
        with open(f"{self.data_dir}/val.txt", "r") as f:
            val_sequences = f.read().splitlines()

        logger.info("Tokenizing sequences...")
        tokenized_train = [
            self.tokenizer.encode(sentence) for sentence in train_sequences
        ]
        tokenized_val = [self.tokenizer.encode(sentence) for sentence in val_sequences]

        # Save tokenized data
        with open(tokenized_train_path, "wb") as file:
            pickle.dump(tokenized_train, file)
        with open(tokenized_val_path, "wb") as file:
            pickle.dump(tokenized_val, file)

    def setup(self, stage=None):
        self.train = TextDataset(self.data_dir / "train_tokenized.pkl")
        self.val = TextDataset(self.data_dir / "val_tokenized.pkl")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
        )


class GPT2Model(LightningModule):
    def __init__(
        self,
        model_name: str,
        num_layers: int,
        num_heads: int,
        dim_embedding: int,
        pad_token_id: int,
        learning_rate: float,
        vocab_size: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=dim_embedding,
            n_layer=num_layers,
            n_head=num_heads,
            pad_token_id=pad_token_id,
        )
        self.model = GPT2LMHeadModel(config)
        self.learning_rate = learning_rate
        self.validation_step_outputs: list[dict] = []
        self.train_step_outputs: list[dict] = []

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        try:
            output: CausalLMOutputWithCrossAttentions = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        except ValueError as e:
            # Error can arise here if there is a batch size of 0 because
            # the number of validation sequences perfectly divides the batch size
            raise ValueError(
                f"Error in forward with input_ids: {input_ids}, attention_mask: {attention_mask}, labels: {labels}"
            ) from e
        return output.loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.FloatTensor:
        loss = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.train_step_outputs.append({"train_loss": loss})
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(
            [x["train_loss"] for x in self.train_step_outputs]
        ).mean()
        self.log("train_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.train_step_outputs = []

    def validation_step(self, batch: dict, batch_idx: int) -> torch.FloatTensor:
        loss = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append({"val_loss": loss})
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def load_config() -> TransformerConfig:
    """Load the config file and return a TransformerConfig object.

    Returns:
        The config object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to the yaml config file")
    args = parser.parse_args()

    # Load the config yaml file
    try:
        with (Path.cwd() / "configs" / f"{args.config_file}.yaml").open("r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error("Config file not found. Please provide a valid yaml file.")
        sys.exit(1)
    return TransformerConfig(**config)


def setup_wandb(config: TransformerConfig) -> WandbLogger | None:
    """Setup wandb if use_wandb is True. Else return None.

    Args:
        config: The config object.

    Returns:
        The wandb logger object or None.
    """
    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.name,
            resume="allow",
        )
        return WandbLogger(experiment=wandb.run)
    return None


def collate_fn(batch, pad_token_id=0):
    max_length = max(len(data["input_ids"]) for data in batch)
    padded_inputs = []
    attention_masks = []
    labels = []
    for data in batch:
        padding_length = max_length - len(data["input_ids"])
        padded_input = data["input_ids"] + [pad_token_id] * padding_length
        attention_mask = data["attention_mask"] + [0] * padding_length
        padded_inputs.append(padded_input)
        attention_masks.append(attention_mask)
        labels.append(padded_input)
    return {
        "input_ids": torch.tensor(padded_inputs),
        "attention_mask": torch.tensor(attention_masks),
        "labels": torch.tensor(labels),
    }


def main(config: TransformerConfig):
    torch.set_float32_matmul_precision("medium")
    devices = find_usable_cuda_devices()
    num_devices = len(devices)
    wandb_logger = setup_wandb(config)

    # Set up checkpoint
    ckpt_dir = Path.cwd() / "checkpoints" / config.name
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{step}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    last_checkpoint = Path(ckpt_dir) / "last.ckpt"
    resume_checkpoint = last_checkpoint if last_checkpoint.exists() else None

    # Instantiate the data module and the model
    batch_size = config.batch_size_per_device * num_devices
    data_dir = Path.cwd() / "data"
    data_module = DataModule(
        data_dir=data_dir, batch_size=batch_size, num_devices=num_devices
    )
    data_module.prepare_data()
    model = GPT2Model(
        config.name,
        config.num_layers,
        config.num_heads,
        config.dim_embedding,
        data_module.tokenizer.pad_token_id,
        config.learning_rate,
        vocab_size=len(data_module.tokenizer.word_to_id),
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
    config = load_config()
    main(config)
