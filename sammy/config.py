from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_PATH = BASE_DIR.parent / "backend" / "models" / "weights.pt"
