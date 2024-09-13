import logging
import sys
import yaml

from pydantic import BaseModel, Field
from pytorch_lightning.loggers import WandbLogger

from planetworldmodel.setting import CONFIG_DIR


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


def load_config(config_file: str, logger: logging.Logger) -> TransformerConfig:
    """Load the config file and return a TransformerConfig object.

    Args:
        config_file: Name of the config file without extension.

    Returns:
        The config object.
    """
    config_path = CONFIG_DIR / f"{config_file}.yaml"
    if not config_path.exists():
        logger.error(f"Config file {config_path} does not exist.")
        sys.exit(1)

    with config_path.open("r") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    return TransformerConfig(**config_dict)


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
