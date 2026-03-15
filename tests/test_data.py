import numpy as np
import torch
import pytest

from ml.preprocess import extract_label, build_label_map, encode_labels, audio_to_mel
from ml.dataset import RadioSignalDataset


CORRUPTED = [
    "0256aa0ac353680c35899c806079337fGliese_163_c",
    "8b690fa7cf57d4f2dba2230b7a078060HD_216520_c",
    "89858af590772f62e7b729f6dd558e95K2-332b",
    "0256aa0ac353680c35899c806079337fGliese_163_c",
]


def test_extract_label_strips_hash():
    assert extract_label("0256aa0ac353680c35899c806079337fGliese_163_c") == "Gliese_163_c"


def test_extract_label_no_hash_passthrough():
    assert extract_label("Kepler-186f") == "Kepler-186f"


def test_build_label_map_unique_sorted():
    labels = np.array(CORRUPTED)
    lm = build_label_map(labels)
    assert set(lm.keys()) == {"Gliese_163_c", "HD_216520_c", "K2-332b"}
    assert list(lm.keys()) == sorted(lm.keys())
    assert list(lm.values()) == list(range(len(lm)))


def test_encode_labels_correct_indices():
    labels = np.array(CORRUPTED)
    lm = build_label_map(labels)
    encoded = encode_labels(labels, lm)
    assert encoded.dtype == np.int64
    assert encoded[0] == encoded[3]
    assert len(set(encoded)) == 3


def test_audio_to_mel_shape():
    waveform = np.random.randn(80000).astype(np.float32)
    mel = audio_to_mel(waveform, n_mels=64)
    assert mel.shape[0] == 1
    assert mel.shape[1] == 64


def test_audio_to_mel_pads_short_signal():
    waveform = np.random.randn(40000).astype(np.float32)
    mel = audio_to_mel(waveform, n_mels=64)
    assert mel.shape[1] == 64


def test_audio_to_mel_truncates_long_signal():
    waveform = np.random.randn(160000).astype(np.float32)
    mel64 = audio_to_mel(waveform, n_mels=64)
    mel32 = audio_to_mel(waveform, n_mels=32)
    assert mel64.shape[1] == 64
    assert mel32.shape[1] == 32


def test_audio_to_mel_normalized():
    waveform = np.random.randn(80000).astype(np.float32)
    mel = audio_to_mel(waveform)
    assert abs(float(mel.mean())) < 0.1
    assert abs(float(mel.std()) - 1.0) < 0.1


def test_dataset_getitem_shape():
    X = np.random.randn(8, 80000, 1).astype(np.float32)
    y = np.array(CORRUPTED[:4].tolist() * 2) if hasattr(CORRUPTED, 'tolist') else np.array(CORRUPTED * 2)
    lm = build_label_map(y)
    ds = RadioSignalDataset(X, y, lm, n_mels=64, augment=False)
    mel, label = ds[0]
    assert mel.shape == (1, 64, 157)
    assert label.dtype == torch.long
