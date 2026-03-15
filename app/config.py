from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

SECRET_KEY = "predprof-secret-key-2025-aliens-x"
ALGORITHM = "HS256"
SESSION_COOKIE = "session_token"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8

DATABASE_URL = f"sqlite:///{BASE_DIR / 'predprof.db'}"

SAVED_MODELS_DIR = BASE_DIR / "saved_models"
TRAINING_LOGS_DIR = BASE_DIR / "training_logs"
DATA_DIR = BASE_DIR / "Data"

MODEL_PATH = SAVED_MODELS_DIR / "model.pth"
CHECKPOINT_PATH = SAVED_MODELS_DIR / "model_checkpoint.pth"
LABEL_MAP_PATH = SAVED_MODELS_DIR / "label_map.json"
HISTORY_PATH = TRAINING_LOGS_DIR / "history.json"
