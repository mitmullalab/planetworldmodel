import argparse
import os
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.loggers import WandbLogger

# Set parser
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="tmp")
args = parser.parse_args()

PAD_TOKEN_ID = 0
NUM_STATES = 10


class SimpleTokenizer:
    def __init__(self, sequences):
        self.pad_token_id = PAD_TOKEN_ID
        word_to_id = {"L": 1, "O": 2, "R": 3, "<pad>": self.pad_token_id}
        next_id = 4
        for state in range(NUM_STATES):
            word_to_id[str(state)] = next_id
            next_id += 1
        self.word_to_id = word_to_id
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
        print(self.word_to_id)

    def tokenize(self, text):
        return text.split()

    def encode(self, sentence):
        return [
            self.word_to_id.get(word, self.word_to_id["<pad>"])
            for word in sentence.split()
        ]

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        if token_ids.ndim == 0:
            token_ids = np.array([token_ids])
        return " ".join(
            self.id_to_word[id] for id in token_ids if id != self.pad_token_id
        )


class RealTextDataset(Dataset):
    def __init__(self, load_file):
        with open(load_file, "rb") as f:
            self.tokenized_sentences = pickle.load(f)

    def __len__(self):
        return len(self.tokenized_sentences)

    def __getitem__(self, idx):
        token_ids = self.tokenized_sentences[idx]
        attention_mask = [1] * len(token_ids)
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": token_ids,
        }


class GPT2Model(LightningModule):
    def __init__(
        self, model_name, tokenizer, vocab_size=50265, n_embd=128, n_layer=12, n_head=4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            pad_token_id=tokenizer.pad_token_id,
        )
        self.model = GPT2LMHeadModel(config)
        self.tokenizer = tokenizer
        self.validation_step_outputs = []
        self.train_step_outputs = []

    def forward(self, input_ids, attention_mask=None, labels=None):
        try:
            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        except ValueError:
            # Error can arise here if there is a batch size of 0 because
            # the number of validation sequences perfectly divides the batch size
            raise ValueError(
                f"Error in forward with input_ids: {input_ids}, attention_mask: {attention_mask}, labels: {labels}"
            )
        return output.loss

    def training_step(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx):
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer

    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.trainer.datamodule.val_dataloader()


def collate_fn(batch):
    max_length = max(len(data["input_ids"]) for data in batch)
    padded_inputs = []
    attention_masks = []
    labels = []
    for data in batch:
        padding_length = max_length - len(data["input_ids"])
        padded_input = data["input_ids"] + [PAD_TOKEN_ID] * padding_length
        attention_mask = data["attention_mask"] + [0] * padding_length
        padded_inputs.append(padded_input)
        attention_masks.append(attention_mask)
        labels.append(padded_input)
    return {
        "input_ids": torch.tensor(padded_inputs),
        "attention_mask": torch.tensor(attention_masks),
        "labels": torch.tensor(labels),
    }


class DataModule(LightningDataModule):
    def __init__(self, model_dir, batch_size=4):
        super().__init__()
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_shards = len(find_usable_cuda_devices())
        data_dir = os.path.join(
            os.path.expanduser("~"),
            f"emergence/data/gridworld/{NUM_STATES}-states-99-length-10000-examples",
        )
        self.data_dir = data_dir

    # def collate_fn(self, batch):
    #   max_length = max(len(data['input_ids']) for data in batch)
    #   padded_inputs = []
    #   attention_masks = []
    #   labels = []

    #   for data in batch:
    #     padding_length = max_length - len(data['input_ids'])
    #     padded_input = data['input_ids'] + [self.tokenizer.pad_token_id] * padding_length
    #     attention_mask = data['attention_mask'] + [0] * padding_length
    #     padded_inputs.append(padded_input)
    #     attention_masks.append(attention_mask)
    #     labels.append(padded_input)

    #   return {
    #     'input_ids': torch.tensor(padded_inputs),
    #     'attention_mask': torch.tensor(attention_masks),
    #     'labels': torch.tensor(labels)
    #   }

    def prepare_data(self, stage=None):
        if Path(f"{self.data_dir}/tokenizer.pt").exists():
            print("Loading existing tokenizer...")
            self.tokenizer = torch.load(f"{self.data_dir}/tokenizer.pt")
            print("...done!")
        else:
            print("Loading datasets...")
            Path(self.data_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{self.data_dir}/train.txt", "r") as f:
                train_sequences = f.read().splitlines()
            with open(f"{self.data_dir}/valid.txt", "r") as f:
                valid_sequences = f.read().splitlines()
            tokenizer = SimpleTokenizer(train_sequences + valid_sequences)
            print("...done!")

            print("Tokenizing sequences...")
            self.tokenizer = tokenizer
            tokenized_train = [
                tokenizer.encode(sentence) for sentence in train_sequences
            ]
            tokenized_valid = [
                tokenizer.encode(sentence) for sentence in valid_sequences
            ]
            print("...done!")

            # Save binary
            with open(f"{self.data_dir}/binary_valid.pkl", "wb") as file:
                pickle.dump(tokenized_valid, file)

            # Save binary
            with open(f"{self.data_dir}/binary_train.pkl", "wb") as file:
                pickle.dump(tokenized_train, file)

            torch.save(self.tokenizer, f"{self.data_dir}/tokenizer.pt")
            print("...done!")

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        load_file = f"{self.data_dir}/binary_train.pkl"
        ds = RealTextDataset(load_file)
        print(f"Created train dataloader from {load_file}")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        load_file = f"{self.data_dir}/binary_valid.pkl"
        ds = RealTextDataset(load_file)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )


def main():
    torch.set_float32_matmul_precision("medium")
    num_gpus = find_usable_cuda_devices()

    num_layers = 8
    n_embd = 512
    n_head = 8
    batch_size_per_gpu = 6
    # save_every = 1000
    batch_size = batch_size_per_gpu * len(num_gpus)

    model_name = args.name
    max_epochs = 500
    use_wandb = True
    if use_wandb:
        wandb_logger = WandbLogger(
            log_model=None, project="emergence", entity="keyonvafa", name=model_name
        )
    model_dir = os.path.join(
        os.path.expanduser("~"), "emergence/checkpoints", model_name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{step}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    last_checkpoint = f"{model_dir}/last.ckpt"
    resume_checkpoint = last_checkpoint if Path(last_checkpoint).exists() else None

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        # devices=1,
        devices=num_gpus,
        logger=wandb_logger if use_wandb else None,
        # val_check_interval=save_every,
        use_distributed_sampler=False,  # NOTE: ADDED NOW
    )

    # Instantiate model and datamodule
    data_module = DataModule(model_dir, batch_size=batch_size)
    data_module.prepare_data()
    model = GPT2Model(
        model_name,
        data_module.tokenizer,
        vocab_size=len(data_module.tokenizer.word_to_id),
        n_embd=n_embd,
        n_layer=num_layers,
        n_head=n_head,
    )
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)


if __name__ == "__main__":
    main()
