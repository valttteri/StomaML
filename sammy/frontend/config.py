import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = str((BASE_DIR.parents[1] / "backend" / "models" / "weights.pt").resolve())
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)
