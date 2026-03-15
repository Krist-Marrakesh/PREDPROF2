import numpy as np
import torch
from torch.utils.data import Dataset

from ml.preprocess import audio_to_mel, encode_labels, build_label_map, save_label_map


class RadioSignalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, label_map: dict, n_mels: int = 64, augment: bool = False):
        self.X = X
        self.labels = encode_labels(y, label_map)
        self.n_mels = n_mels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        mel = audio_to_mel(self.X[idx], n_mels=self.n_mels)
        if self.augment:
            mel = self._augment(mel)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label

    def _augment(self, mel: torch.Tensor) -> torch.Tensor:
        _, n_mels, time_steps = mel.shape

        if torch.rand(1).item() < 0.5:
            t = torch.randint(0, time_steps // 4, (1,)).item()
            t0 = torch.randint(0, time_steps - t, (1,)).item()
            mel = mel.clone()
            mel[:, :, t0:t0 + t] = 0.0

        if torch.rand(1).item() < 0.5:
            f = torch.randint(0, n_mels // 4, (1,)).item()
            f0 = torch.randint(0, n_mels - f, (1,)).item()
            mel = mel.clone()
            mel[:, f0:f0 + f, :] = 0.0

        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(mel) * 0.05
            mel = mel + noise

        return mel


def build_datasets(train_x, train_y, valid_x, valid_y, n_mels: int = 64):
    label_map = build_label_map(np.concatenate([train_y, valid_y]))
    save_label_map(label_map)
    train_ds = RadioSignalDataset(train_x, train_y, label_map, n_mels=n_mels, augment=True)
    valid_ds = RadioSignalDataset(valid_x, valid_y, label_map, n_mels=n_mels, augment=False)
    return train_ds, valid_ds, label_map
