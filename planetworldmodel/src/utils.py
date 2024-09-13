import logging
import sys
import yaml

from pydantic import BaseModel, Field

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
