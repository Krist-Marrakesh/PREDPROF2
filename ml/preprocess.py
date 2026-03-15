import re
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T


SAMPLE_RATE = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
MAX_LENGTH = 80000
LABEL_MAP_PATH = Path(__file__).parent.parent / "saved_models" / "label_map.json"


def extract_label(corrupted: str) -> str:
    match = re.match(r"^[0-9a-f]{32}(.+)$", corrupted)
    return match.group(1) if match else corrupted


def build_label_map(labels: np.ndarray) -> dict:
    unique = sorted(set(extract_label(l) for l in labels))
    return {name: idx for idx, name in enumerate(unique)}


def save_label_map(label_map: dict, path: Path = LABEL_MAP_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)


def load_label_map(path: Path = LABEL_MAP_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def encode_labels(labels: np.ndarray, label_map: dict) -> np.ndarray:
    return np.array([label_map[extract_label(l)] for l in labels], dtype=np.int64)


def extract_active_region(waveform: np.ndarray, window: int = 512, threshold: float = 1e-4) -> np.ndarray:
    rms = np.sqrt(np.convolve(waveform ** 2, np.ones(window) / window, mode="same"))
    active = np.where(rms > threshold)[0]
    if len(active) == 0:
        return waveform
    start = max(0, active[0] - window)
    end = min(len(waveform), active[-1] + window)
    return waveform[start:end]


def audio_to_mel(waveform: np.ndarray, n_mels: int = N_MELS) -> torch.Tensor:
    if waveform.ndim > 1:
        waveform = waveform.squeeze(-1)

    waveform = extract_active_region(waveform)

    if len(waveform) < MAX_LENGTH:
        pad_left = (MAX_LENGTH - len(waveform)) // 2
        pad_right = MAX_LENGTH - len(waveform) - pad_left
        waveform = np.pad(waveform, (pad_left, pad_right))
    else:
        waveform = waveform[:MAX_LENGTH]

    wav_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        power=2.0,
    )
    db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)

    mel = mel_transform(wav_tensor)
    mel_db = db_transform(mel)

    mean = mel_db.mean()
    std = mel_db.std() + 1e-6
    return (mel_db - mean) / std
