import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CHECKPOINT_PATH, LABEL_MAP_PATH, HISTORY_PATH
from ml.model import build_model
from ml.preprocess import audio_to_mel, load_label_map

_model: Optional[torch.nn.Module] = None
_label_map: Optional[dict] = None
_inv_label_map: Optional[dict] = None
_model_params: Optional[dict] = None
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _load_model():
    global _model, _label_map, _inv_label_map, _model_params
    if not CHECKPOINT_PATH.exists():
        return False
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    _label_map = checkpoint["label_map"]
    _model_params = checkpoint.get("best_params", {})
    num_classes = checkpoint.get("num_classes", len(_label_map))
    _model = build_model(
        num_classes=num_classes,
        n_mels=_model_params.get("n_mels", 64),
        dropout_p1=_model_params.get("dropout_p1", 0.25),
        dropout_p2=_model_params.get("dropout_p2", 0.25),
        dropout_p3=_model_params.get("dropout_p3", 0.5),
        num_conv_blocks=_model_params.get("num_conv_blocks", 4),
    )
    _model.load_state_dict(checkpoint["state_dict"])
    _model.to(DEVICE)
    _model.eval()
    _inv_label_map = {v: k for k, v in _label_map.items()}
    return True


def is_model_ready() -> bool:
    if _model is None:
        _load_model()
    return _model is not None


def predict_batch(X: np.ndarray) -> list[dict]:
    if not is_model_ready():
        raise RuntimeError("Model not loaded")
    n_mels = _model_params.get("n_mels", 64)
    results = []
    with torch.inference_mode():
        for i, waveform in enumerate(X):
            mel = audio_to_mel(waveform, n_mels=n_mels).unsqueeze(0).to(DEVICE)
            logits = _model(mel)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_idx = int(probs.argmax())
            results.append({
                "index": i,
                "predicted_class": _inv_label_map[pred_idx],
                "confidence": float(probs[pred_idx]),
                "top3": [
                    {"class": _inv_label_map[int(j)], "prob": float(probs[j])}
                    for j in probs.argsort()[::-1][:3]
                ],
            })
    return results


def get_training_history() -> Optional[dict]:
    if not HISTORY_PATH.exists():
        return None
    with open(HISTORY_PATH) as f:
        return json.load(f)


def get_model_info() -> dict:
    if not is_model_ready():
        return {"ready": False}
    history = get_training_history()
    return {
        "ready": True,
        "best_val_acc": history["best_val_acc"] if history else None,
        "num_classes": len(_label_map),
        "params": _model_params,
    }
