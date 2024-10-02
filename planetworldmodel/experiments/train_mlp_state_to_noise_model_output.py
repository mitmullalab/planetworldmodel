import argparse
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import tqdm
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from planetworldmodel import setup_wandb
from planetworldmodel.setting import CKPT_DIR, DATA_DIR, GEN_DATA_DIR


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPConfig:
    def __init__(self):
        self.name = "mlp_model_output_predictor"
        self.batch_size = 1_024
        self.num_workers = 4
        self.learning_rate = 0.01
        self.max_epochs = 10
        self.hidden_dims = [32,32]
        self.store_predictions = False
        self.prediction_path = "mlp_state_to_transformer_output_predictions"
        self.wandb_project = "MLP-state-to-transformer-output"
        self.wandb_entity = "petergchang"
        self.use_wandb = True


class FlattenedStateDataset(Dataset):
    def __init__(
        self, 
        input_file_path: Path, 
        target_file_path: Path,
        num_data: int | None = None,
        seed: int = 0,
        target_dim_idx: int | None = None,
    ):
        inputs = np.load(input_file_path)
        targets = np.load(target_file_path)

        # Flatten the data
        self.input_states = torch.tensor(inputs.reshape(-1, inputs.shape[-1]), dtype=torch.float32)
        self.target_states = torch.tensor(targets.reshape(-1, targets.shape[-1]), dtype=torch.float32)
        
        if target_dim_idx is not None:
            self.target_states = self.target_states[:, target_dim_idx:target_dim_idx+1]
        
        # Subsample the data
        if num_data is not None:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.input_states), num_data, replace=False)
            self.input_states = self.input_states[indices]
            self.target_states = self.target_states[indices]

    def __len__(self) -> int:
        return len(self.input_states)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_sequence": self.input_states[idx],
            "target_sequence": self.target_states[idx],
        }
        
class StatePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(StatePredictor, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

class StatePredictionDataModule(LightningDataModule):
    def __init__(
        self, 
        data_dir: Path, 
        pretrained: Literal["sp", "ntp", "sp_direct"],
        num_data_points: int,
        batch_size: int = 1024, 
        num_workers: int = 4,
        target_dim_idx: int | None = None,
        checkpoint_epoch_num: int | None = None
    ):
        super().__init__()
        epoch_suffix = "" if checkpoint_epoch_num is None else f"_epoch_{checkpoint_epoch_num}"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_file = self.data_dir / "standardized_state_train_heavier_fixed.npy"
        self.train_target_file = self.data_dir / f"{pretrained}_noise_trained_model_predictions_train{epoch_suffix}_data_points_{num_data_points}.npy"
        self.val_file = self.data_dir / "standardized_state_val_heavier_fixed.npy"
        self.val_target_file = self.data_dir / f"{pretrained}_noise_trained_model_predictions_val{epoch_suffix}_data_points_{num_data_points}.npy"
        self.target_dim_idx = target_dim_idx

    def setup(self, stage=None):
        self.train = FlattenedStateDataset(
            self.train_file, self.train_target_file, num_data=1_000_000, target_dim_idx=self.target_dim_idx
        )
        self.val = FlattenedStateDataset(
            self.val_file, self.val_target_file, num_data=100_000, target_dim_idx=self.target_dim_idx
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class MLPRegressor(LightningModule):
    def __init__(self, config: MLPConfig, input_dim: int, output_dim: int, l1_lambda: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.output_dim = output_dim
        self.l1_lambda = l1_lambda
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        self.store_predictions = config.store_predictions
        self.prediction_path = config.prediction_path
        self.inputs = []
        self.predictions = []
        self.targets = []
        
        self.best_val_r2 = -float('inf')
        self.best_val_r2_per_dim = [-float('inf')] * output_dim
        self.val_predictions = []
        self.val_targets = []
        

    def forward(self, x):
        representation = self.hidden_layers(x)
        output = self.output_layer(representation)
        return output, representation

    def training_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        target_sequence = batch["target_sequence"]
        predictions, representation = self(input_sequence)
        mse_loss = F.mse_loss(predictions, target_sequence)
        l1_loss = self.l1_lambda * torch.norm(representation, 1)
        loss = mse_loss + l1_loss
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        if self.store_predictions:
            self.predictions.append(predictions.detach().cpu().numpy())
            self.inputs.append(input_sequence.detach().cpu().numpy())
            self.targets.append(target_sequence.detach().cpu().numpy())
        return loss

    def on_train_epoch_end(self):
        if not self.store_predictions:
            return

        epoch_predictions = np.concatenate(self.predictions)
        epoch_inputs = np.concatenate(self.inputs)
        epoch_targets = np.concatenate(self.targets)

        prediction_path = GEN_DATA_DIR / self.prediction_path
        prediction_path.mkdir(exist_ok=True, parents=True)
        np.save(prediction_path / f"mlp_predictions_epoch_{self.current_epoch}.npy", epoch_predictions)
        np.save(prediction_path / f"mlp_inputs_epoch_{self.current_epoch}.npy", epoch_inputs)
        np.save(prediction_path / f"mlp_targets_epoch_{self.current_epoch}.npy", epoch_targets)

        self.predictions, self.inputs, self.targets = [], [], []

    def validation_step(self, batch, batch_idx):
        input_sequence = batch["input_sequence"]
        target_sequence = batch["target_sequence"]
        predictions, representation = self(input_sequence)
        mse_loss = F.mse_loss(predictions, target_sequence)
        l1_loss = self.l1_lambda * torch.norm(representation, 1)
        loss = mse_loss + l1_loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            
        # Store predictions and targets for R2 calculation
        self.val_predictions.append(predictions.detach())
        self.val_targets.append(target_sequence.detach())
        
        return loss
    
    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.val_predictions)
        all_targets = torch.cat(self.val_targets)
        
        # Calculate R2 score
        var_y = torch.var(all_targets)
        mse = torch.mean((all_preds - all_targets) ** 2)
        r2 = 1 - mse / var_y
        
        self.log("val_r2", r2, prog_bar=True, sync_dist=True)
        
        if r2 > self.best_val_r2:
            self.best_val_r2 = r2
        self.log("best_val_r2", self.best_val_r2, prog_bar=True, sync_dist=True)
        
        
        # Calculate R2 score for each dimension
        for dim in range(self.output_dim):
            var_y_dim = torch.var(all_targets[:, dim])
            mse_dim = torch.mean((all_preds[:, dim] - all_targets[:, dim]) ** 2)
            r2_dim = 1 - mse_dim / var_y_dim
            
            self.log(f"val_r2_dim_{dim}", r2_dim, prog_bar=False, sync_dist=True)
            
            if r2_dim > self.best_val_r2_per_dim[dim]:
                self.best_val_r2_per_dim[dim] = r2_dim
            self.log(f"best_val_r2_dim_{dim}", self.best_val_r2_per_dim[dim], prog_bar=False, sync_dist=True)
        
        # Clear stored predictions and targets
        self.val_predictions.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
    
    
def get_representations(model, data_module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    train_representations, val_representations = [], []
    train_states, val_states = [], []

    with torch.no_grad():
        for loader, representations, states in [
            (data_module.train_dataloader(), train_representations, train_states),
            (data_module.val_dataloader(), val_representations, val_states)
        ]:
            for batch in tqdm.tqdm(loader, desc="Extracting representations"):
                input_sequence = batch["input_sequence"].to(device)
                _, representation = model(input_sequence)
                representations.append(representation.cpu())
                states.append(batch["input_sequence"].cpu())

    return (torch.cat(train_representations), torch.cat(val_representations),
            torch.cat(train_states), torch.cat(val_states))



def train_state_predictor(train_representations, train_states, val_representations, val_states, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_representations.shape[1]
    output_dim = train_states.shape[1]
    hidden_dim = 128
    num_layers = 1

    state_predictor = StatePredictor(input_dim, hidden_dim, output_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(state_predictor.parameters(), lr=0.001)

    train_dataset = TensorDataset(train_representations, train_states)
    val_dataset = TensorDataset(val_representations, val_states)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=True)

    best_val_r2 = float('-inf')
    for epoch in range(config.max_epochs):
        state_predictor.train()
        for reps, states in train_loader:
            reps, states = reps.to(device), states.to(device)
            optimizer.zero_grad()
            predicted_states = state_predictor(reps)
            loss = F.mse_loss(predicted_states, states.float())
            loss.backward()
            optimizer.step()

        state_predictor.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for reps, states in val_loader:
                reps, states = reps.to(device), states.to(device)
                predicted_states = state_predictor(reps)
                val_preds.append(predicted_states)
                val_true.append(states)

        val_preds = torch.cat(val_preds)
        val_true = torch.cat(val_true)
        mse_per_dim = torch.mean((val_preds - val_true.float()) ** 2, 0)
        var_per_dim = torch.var(val_true.float(), 0)
        r2_per_dim = 1 - mse_per_dim / var_per_dim
        val_r2 = torch.mean(r2_per_dim)
        print(f"Epoch {epoch+1}, Validation R2: {val_r2.item():.4f}")
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2

    return best_val_r2.item()


def l1_grid_search(config: MLPConfig, data_module, input_dim: int, output_dim: int, l1_values: list[float]):
    best_val_loss = float('inf')
    best_l1_lambda = None
    best_model = None

    for l1_lambda in l1_values:
        model = MLPRegressor(config, input_dim, output_dim, l1_lambda=l1_lambda)
        
        trainer = Trainer(
            max_epochs=config.max_epochs,
            logger=False,  # Disable logging for grid search
            accelerator="auto",
            devices=1,
            strategy="auto",
        )
        
        trainer.fit(model, data_module)
        
        # Get the best validation loss
        val_loss = trainer.callback_metrics['val_loss'].item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_l1_lambda = l1_lambda
            best_model = model

    return best_model, best_l1_lambda, best_val_loss


def main(config: MLPConfig, args):
    torch.set_float32_matmul_precision("medium")
    devices = find_usable_cuda_devices()
    wandb_logger = setup_wandb(config)

    data_dir = DATA_DIR / "two_body_problem"
    
    # # standardize the state dataset
    # if not (data_dir / "standardized_state_train_heavier_fixed.npy").exists() \
    #     or not (data_dir / "standardized_state_val_heavier_fixed.npy").exists():
    #     train_states = np.load(data_dir / "state_train_heavier_fixed.npy")
    #     val_states = np.load(data_dir / "state_val_heavier_fixed.npy")
    #     train_states_mean = train_states.mean(axis=(0,1))
    #     train_states_std = train_states.std(axis=(0,1))
    #     train_states = (train_states - train_states_mean) / train_states_std
    #     val_states = (val_states - train_states_mean) / train_states_std
    #     np.save(data_dir / "standardized_state_train_heavier_fixed.npy", train_states)
    #     np.save(data_dir / "standardized_state_val_heavier_fixed.npy", val_states)

    # # standardize the model prediction dataset
    # if not (data_dir / f"standardized_{pretrained}_noise_trained_model_predictions_train.npy").exists() \
    #     or not (data_dir / f"standardized_{pretrained}_noise_trained_model_predictions_val.npy").exists():
    #     train_output = np.load(data_dir / f"{pretrained}_noise_trained_model_predictions_train.npy")
    #     val_output = np.load(data_dir / f"{pretrained}_noise_trained_model_predictions_val.npy")
    #     train_output_mean = train_output.mean(axis=(0,1))
    #     train_output_std = np.array(torch.from_numpy(train_output).std(axis=(0,1)))  # Weird numpy bug
    #     standardized_train_output = (train_output - train_output_mean) / train_output_std
    #     standardized_val_states = (val_output - train_output_mean) / train_output_std
    #     breakpoint()
    #     np.save(data_dir / f"standardized_{pretrained}_noise_trained_model_predictions_train.npy", standardized_train_output)
    #     np.save(data_dir /f"standardized_{pretrained}_noise_trained_model_predictions_val.npy", standardized_val_states)
    
    l1_values = [0.0, 0.0001, 0.1, 1.0]
    
    best_r2s_mlp1, best_r2s_mlp2 = [], []
    result_dir = Path.cwd() / "mlp_state_to_transformer_output_predictions"
    result_dir.mkdir(exist_ok=True, parents=True)
    if args.multi_task_learner:
        data_module = StatePredictionDataModule(
            data_dir=data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pretrained=args.pretrained,
            num_data_points=args.num_data_points,
            target_dim_idx=None,
            checkpoint_epoch_num=args.checkpoint_epoch_num,
        )
        
        data_module.setup()
        sample = data_module.train[0]
        input_dim = sample["input_sequence"].shape[-1]
        output_dim = sample["target_sequence"].shape[-1]
        
        # # Perform grid search
        # best_model, best_l1_lambda, best_val_loss = l1_grid_search(config, data_module, input_dim, output_dim, l1_values)

        model = MLPRegressor(config, input_dim, output_dim)
        
        ckpt_dir = CKPT_DIR / config.name
        best_model_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            save_top_k=1,
            monitor='val_r2',
            mode='max'
        )

        trainer = Trainer(
            max_epochs=config.max_epochs,
            callbacks=[best_model_callback],
            logger=wandb_logger,
            accelerator="auto",
            devices=devices,
            strategy="auto",
        )
        
        # print('-' * 20 + " First MLP " + '-' * 20)
        trainer.fit(model, data_module)
        
        # Log the best overall R2 score and per-dimension R2 scores
        best_r2_mlp1 = model.best_val_r2
        
        # Extract representations
        train_representations, val_representations, train_states, val_states = get_representations(
            model, data_module
        )
        
        # Train state predictor and get best R2
        best_r2_mlp2 = train_state_predictor(
            train_representations, train_states, val_representations, val_states, config
        )
        print(f"Best state prediction R2: {best_r2_mlp2:.4f}")
        
        # Save result as txt
        file_name = f"{args.pretrained}_noise_best_r2_epoch_{args.checkpoint_epoch_num}_data_points_{args.num_data_points}_multitask"
        with open(result_dir / f"{file_name}.txt", "w") as f:
            f.write(f"Best R2 MLP1: {best_r2_mlp1:.4f}\n")
            f.write(f"Best R2 MLP2: {best_r2_mlp2:.4f}\n")
    else:
        for i in range(args.num_models):
            data_module = StatePredictionDataModule(
                data_dir=data_dir,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pretrained=args.pretrained,
                num_data_points=args.num_data_points,
                target_dim_idx=i,
                checkpoint_epoch_num=args.checkpoint_epoch_num,
            )
            
            data_module.setup()
            sample = data_module.train[0]
            input_dim = sample["input_sequence"].shape[-1]
            output_dim = sample["target_sequence"].shape[-1]

            model = MLPRegressor(config, input_dim, output_dim)

            ckpt_dir = CKPT_DIR / config.name
            best_model_callback = ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best",
                save_top_k=1,
                monitor='val_r2',
                mode='max'
            )

            trainer = Trainer(
                max_epochs=config.max_epochs,
                callbacks=[best_model_callback],
                logger=wandb_logger,
                accelerator="auto",
                devices=devices,
                strategy="auto",
            )
            
            # print('-' * 20 + " First MLP " + '-' * 20)
            trainer.fit(model, data_module)
            
            # Log the best overall R2 score and per-dimension R2 scores
            best_r2 = model.best_val_r2
            best_r2s_mlp1.append(best_r2.item())
            # print(f"Best overall validation R2 score: {best_r2:.4f}")
            # if config.use_wandb:
            #     wandb.log({"best_val_r2": best_r2})
                
            # for dim in range(output_dim):
            #     best_r2_dim = model.best_val_r2_per_dim[dim]
            #     print(f"Best validation R2 score for dimension {dim}: {best_r2_dim:.4f}")
            #     if config.use_wandb:
            #         wandb.log({f"best_val_r2_dim_{dim}": best_r2_dim})
                    
            # print('-' * 20 + " Second MLP " + '-' * 20)
            # Extract representations
            train_representations, val_representations, train_states, val_states = get_representations(
                model, data_module
            )
            
            # Train state predictor and get best R2
            best_r2 = train_state_predictor(
                train_representations, train_states, val_representations, val_states, config
            )
            best_r2s_mlp2.append(best_r2)
            print(f"Best state prediction R2: {best_r2:.4f}")
            if config.use_wandb:
                wandb.log({"best_state_prediction_r2": best_r2})

        best_r2s_mlp1, best_r2s_mlp2 = np.array(best_r2s_mlp1), np.array(best_r2s_mlp2)
        print('-' * 20 + " MLP1 " + '-' * 20)
        print(best_r2s_mlp1)
        print(f"Mean best R2: {np.mean(best_r2s_mlp1):.4f}")
        print(f"Std best R2: {np.std(best_r2s_mlp1):.4f}")
        # Save result as txt
        file_name = f"{args.pretrained}_noise_best_r2s_epoch_{args.checkpoint_epoch_num}_data_points_{args.num_data_points}"
        with open(result_dir / f"{file_name}_mlp1.txt", "w") as f:
            f.write(f"Mean best R2: {np.mean(best_r2s_mlp1):.4f}\n")
            f.write(f"Std best R2: {np.std(best_r2s_mlp1):.4f}\n")
            f.write(f"Best R2s: {best_r2s_mlp1}\n")
            
        print('-' * 20 + " MLP2 " + '-' * 20)
        print(best_r2s_mlp2)
        print(f"Mean best R2: {np.mean(best_r2s_mlp2):.4f}")
        print(f"Std best R2: {np.std(best_r2s_mlp2):.4f}")
        # Save result as txt
        with open(result_dir / f"{file_name}_mlp2.txt", "w") as f:
            f.write(f"Mean best R2: {np.mean(best_r2s_mlp2):.4f}\n")
            f.write(f"Std best R2: {np.std(best_r2s_mlp2):.4f}\n")
            f.write(f"Best R2s: {best_r2s_mlp2}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained",
        type=str,
        choices=["sp", "ntp", "sp_direct"],
    )
    parser.add_argument(
        "--checkpoint_epoch_num",
        type=int,
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--multi_task_learner",
        action="store_true",
    )
    parser.add_argument(
        "--num_data_points",
        type=int,
    )
    args = parser.parse_args()

    config = MLPConfig()

    main(config, args)
    