from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"
CONFIG_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
GEN_DATA_DIR = DATA_DIR / "generated_data"
