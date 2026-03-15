import io
import numpy as np


def parse_upload(content: bytes, filename: str) -> np.ndarray:
    if filename.endswith(".npy"):
        X = np.load(io.BytesIO(content), allow_pickle=True)
    elif filename.endswith(".npz"):
        data = np.load(io.BytesIO(content), allow_pickle=True)
        key = [k for k in data.files if "x" in k.lower()] or list(data.files)
        X = data[key[0]]
    else:
        raise ValueError(f"Unsupported format: {filename}")
    if X.ndim == 1:
        X = X[np.newaxis, :]
    return X
