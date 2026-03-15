import sys
import json
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

import re
import hashlib
from collections import Counter

from ml.preprocess import audio_to_mel, extract_label, save_label_map
from ml.model import build_model

ROOT = Path(__file__).parent.parent
SAVED_MODELS = ROOT / "saved_models"
TRAINING_LOGS = ROOT / "training_logs"
SAVED_MODELS.mkdir(exist_ok=True)
TRAINING_LOGS.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE.type == "mps":
    torch.backends.mps.enable_fallback_kernels = True
NUM_CLASSES = 20

print(f"Device: {DEVICE}")

data = np.load(ROOT / "Data" / "Data(1).npz", allow_pickle=True)
train_x = data["train_x"]
valid_x = data["valid_x"]

raw_train_y = data["train_y"]
raw_valid_y = data["valid_y"]
print(f"Train: {train_x.shape}, Valid: {valid_x.shape}")


def get_hex(label):
    m = re.match(r'^([0-9a-f]{32})', label)
    return m.group(1) if m else None


def restore_labels(raw_labels, n_signals):
    """Восстанавливает правильное соответствие меток сигналам.

    Метки перемешаны практикантами. hex = MD5(within_class_idx + planet).
    Сигналы хранятся блоками по классам (по убыванию кол-ва в классе).
    within_class_idx указывает позицию внутри блока.
    """
    planets = sorted(set(extract_label(l) for l in raw_labels))
    class_counts = Counter(extract_label(l) for l in raw_labels)

    hash_lookup = {}
    for planet in planets:
        for i in range(n_signals):
            h = hashlib.md5(f"{i}{planet}".encode()).hexdigest()
            hash_lookup[h] = (i, planet)

    class_order = []
    remaining = list(planets)
    offset = 0
    for _ in range(len(planets)):
        best_planet, best_matches = None, -1
        for planet in remaining:
            cnt = class_counts[planet]
            matches = sum(
                1 for l in raw_labels
                if extract_label(l) == planet
                and get_hex(l) in hash_lookup
                and hash_lookup[get_hex(l)][0] < cnt
            )
            if matches > best_matches:
                best_matches = matches
                best_planet = planet
        class_order.append(best_planet)
        remaining.remove(best_planet)
        offset += class_counts[best_planet]

    label_map = {name: idx for idx, name in enumerate(sorted(set(planets)))}

    offsets = {}
    pos = 0
    for planet in class_order:
        offsets[planet] = pos
        pos += class_counts[planet]

    confident = {}
    for label in raw_labels:
        hex_val = get_hex(label)
        if hex_val not in hash_lookup:
            continue
        within_idx, planet = hash_lookup[hex_val]
        signal_pos = offsets[planet] + within_idx
        if signal_pos < n_signals:
            if signal_pos not in confident:
                confident[signal_pos] = label_map[planet]
            elif confident[signal_pos] != label_map[planet]:
                del confident[signal_pos]

    labels = np.full(n_signals, -1, dtype=np.int64)
    for pos, cls in confident.items():
        labels[pos] = cls

    pos = 0
    for planet in class_order:
        cnt = class_counts[planet]
        cls = label_map[planet]
        for j in range(cnt):
            if labels[pos + j] == -1:
                labels[pos + j] = cls
        pos += cnt

    return labels, label_map, class_order


print("Restoring label alignment...")
train_y, label_map, train_class_order = restore_labels(raw_train_y, len(train_x))
valid_y, _, valid_class_order = restore_labels(raw_valid_y, len(valid_x))
save_label_map(label_map)
print(f"Classes: {len(label_map)}")
print(f"Train class order: {train_class_order}")
confident_train = (train_y >= 0).sum()
confident_valid = (valid_y >= 0).sum()
print(f"Train confident: {confident_train}/{len(train_x)}, Valid confident: {confident_valid}/{len(valid_x)}")

print("Precomputing spectrograms...")
N_MELS_VARIANTS = [32, 64, 128]
spec_cache: dict = {}
for n in N_MELS_VARIANTS:
    t0 = time.time()
    X_tr = torch.stack([audio_to_mel(s, n_mels=n) for s in train_x])
    y_tr = torch.tensor(train_y, dtype=torch.long)
    X_val = torch.stack([audio_to_mel(s, n_mels=n) for s in valid_x])
    y_val = torch.tensor(valid_y, dtype=torch.long)
    spec_cache[n] = (X_tr, y_tr, X_val, y_val)
    print(f"  n_mels={n}: {time.time()-t0:.1f}s")


def make_loaders(n_mels, batch_size):
    X_tr, y_tr, X_val, y_val = spec_cache[n_mels]
    return (
        DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size * 2, shuffle=False),
    )


def augment_batch(x):
    x = x.clone()
    if random.random() < 0.5:
        x += torch.randn_like(x) * 0.02
    return x


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = augment_batch(x).to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.detach().argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            total_loss += criterion(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


OPTUNA_TRIALS = 40


def objective(trial):
    n_mels = trial.suggest_categorical("n_mels", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout_p1 = trial.suggest_float("dropout_p1", 0.1, 0.4)
    dropout_p2 = trial.suggest_float("dropout_p2", 0.1, 0.4)
    dropout_p3 = trial.suggest_float("dropout_p3", 0.3, 0.6)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    num_conv_blocks = trial.suggest_categorical("num_conv_blocks", [3, 4])
    epochs = trial.suggest_int("epochs", 30, 100)

    train_loader, valid_loader = make_loaders(n_mels, batch_size)
    model = build_model(
        num_classes=NUM_CLASSES, n_mels=n_mels,
        dropout_p1=dropout_p1, dropout_p2=dropout_p2, dropout_p3=dropout_p3,
        num_conv_blocks=num_conv_blocks,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    OptCls = torch.optim.AdamW if optimizer_name == "AdamW" else torch.optim.Adam
    optimizer = OptCls(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_epoch(model, valid_loader, criterion)
        scheduler.step(val_loss)
        best_val_acc = max(best_val_acc, val_acc)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_val_acc


print(f"\nOptuna: {OPTUNA_TRIALS} trials...")
study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=10, interval_steps=2),
    sampler=TPESampler(n_startup_trials=10, multivariate=True, seed=42),
    study_name="radio_signal_clf",
)
t0 = time.time()
study.optimize(objective, n_trials=OPTUNA_TRIALS)
print(f"Optuna done in {time.time()-t0:.0f}s | best val_acc={study.best_value:.4f}")
best = study.best_params
print("Best params:", best)

print("\nFinal training...")
FINAL_EPOCHS = best["epochs"]
train_loader, valid_loader = make_loaders(best["n_mels"], best["batch_size"])
model = build_model(
    num_classes=NUM_CLASSES, n_mels=best["n_mels"],
    dropout_p1=best["dropout_p1"], dropout_p2=best["dropout_p2"],
    dropout_p3=best["dropout_p3"], num_conv_blocks=best["num_conv_blocks"],
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
OptCls = torch.optim.AdamW if best["optimizer"] == "AdamW" else torch.optim.Adam
optimizer = OptCls(model.parameters(), lr=best["lr"], weight_decay=best["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, min_lr=1e-6)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
patience_counter = 0

for epoch in range(1, FINAL_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = eval_epoch(model, valid_loader, criterion)
    scheduler.step(val_loss)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVED_MODELS / "model.pth")
        torch.save({
            "state_dict": model.state_dict(),
            "label_map": label_map,
            "best_params": best,
            "val_acc": val_acc,
            "num_classes": NUM_CLASSES,
        }, SAVED_MODELS / "model_checkpoint.pth")
        patience_counter = 0
        marker = " ★"
    else:
        patience_counter += 1
        marker = ""

    if epoch % 5 == 0 or epoch == 1:
        lr_cur = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{FINAL_EPOCHS} | tr {tr_loss:.4f}/{tr_acc:.4f} | "
              f"val {val_loss:.4f}/{val_acc:.4f} | lr={lr_cur:.2e} | {time.time()-t0:.1f}s{marker}")

    if patience_counter >= 15:
        print(f"Early stop @ {epoch}")
        break

with open(TRAINING_LOGS / "history.json", "w") as f:
    json.dump({"best_params": best, "best_val_acc": best_val_acc,
               "num_classes": NUM_CLASSES, "label_map": label_map, "history": history}, f, indent=2)

import h5py
state = torch.load(SAVED_MODELS / "model.pth", map_location="cpu")
with h5py.File(SAVED_MODELS / "model.h5", "w") as f:
    for k, v in state.items():
        f.create_dataset(k, data=v.numpy())
    f.attrs["label_map"] = json.dumps(label_map)
    f.attrs["best_params"] = json.dumps(best)
    f.attrs["val_acc"] = best_val_acc
    f.attrs["num_classes"] = NUM_CLASSES

print(f"\nDone! Best val_accuracy: {best_val_acc:.4f}")
print(f"Saved: {SAVED_MODELS / 'model.pth'}, {SAVED_MODELS / 'model.h5'}")
