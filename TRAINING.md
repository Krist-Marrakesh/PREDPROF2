# Документация по обучению модели

## 1. Описание задачи

Классификация радиосигналов инопланетных цивилизаций. Модель определяет цивилизацию-отправителя по записи радиосигнала.

- **Количество классов:** 20 (экзопланеты)
- **Обучающая выборка:** 1200 записей
- **Валидационная выборка:** 400 записей
- **Формат сигнала:** wav, 80000 отсчетов, 1 канал, sample rate 22050 Hz

## 2. Восстановление меток

Исходные метки в наборе данных повреждены практикантами: вместо целочисленных значений содержат строки вида `<32-символьный hex-префикс><название_планеты>`.

**Пример повреждённой метки:**
```
0256aa0ac353680c35899c806079337fGliese_163_c
```

**Алгоритм восстановления:**

1. Удаление hex-префикса (32 символа `[0-9a-f]`) регулярным выражением `^[0-9a-f]{32}`
2. Извлечение названия цивилизации (планеты)
3. Сортировка уникальных названий по алфавиту
4. Присвоение целочисленных меток начиная с 0

**Реализация** — `ml/preprocess.py:extract_label()`:
```python
def extract_label(corrupted: str) -> str:
    match = re.match(r"^[0-9a-f]{32}(.+)$", corrupted)
    return match.group(1) if match else corrupted
```

**Восстановленная карта меток** (`saved_models/label_map.json`):

| Класс | Цивилизация (планета) |
|------:|:----------------------|
| 0     | 55_Cancri_Bc          |
| 1     | Gliese_               |
| 2     | Gliese_12_b           |
| 3     | Gliese_163_c          |
| 4     | HD_20794_d            |
| 5     | HD_216520_c           |
| 6     | HIP_38594_b           |
| 7     | K2-155d               |
| 8     | K2-288Bb              |
| 9     | K2-332b               |
| 10    | K2-72e                |
| 11    | Kepler-155c           |
| 12    | Kepler-174d           |
| 13    | Kepler-186f           |
| 14    | Kepler-22b            |
| 15    | Kepler-283c           |
| 16    | Kepler-296e           |
| 17    | Kepler-296f           |
| 18    | Kepler-62e            |
| 19    | Kepler-62f            |

## 3. Предобработка данных

**Модуль:** `ml/preprocess.py`

### Пайплайн преобразования сигнала в Мел-спектрограмму:

1. **Извлечение активной области** (`extract_active_region`) — обрезка тишины по RMS-порогу
2. **Нормализация длины** — pad/truncate до `MAX_LENGTH = 80000` отсчётов
3. **Мел-спектрограмма** (`torchaudio.transforms.MelSpectrogram`):
   - `sample_rate = 22050`
   - `n_fft = 1024`
   - `hop_length = 512`
   - `n_mels` — варьируется (32, 64, 128)
4. **Перевод в децибелы** (`AmplitudeToDB`, `top_db=80`)
5. **Z-нормализация** — `(x - mean) / (std + 1e-6)`

### Аугментация (при обучении):

- Гауссов шум (вероятность 50%, sigma=0.02) — `train.py:augment_batch()`
- Time masking (25% длины, вероятность 50%) — `dataset.py`
- Frequency masking (25% частот, вероятность 50%) — `dataset.py`

## 4. Архитектура модели

**Модуль:** `ml/model.py`

**Класс:** `AudioClassifier` — свёрточная нейронная сеть (CNN).

### Структура:

```
Вход: (batch, 1, n_mels, time_steps)
          │
          ▼
    ┌─────────────────────────────┐
    │ Conv Block 1                │
    │ Conv2d(1→32, 3x3) + BN +   │
    │ ReLU + MaxPool2d(2) +       │
    │ Dropout2d(p1)               │
    ├─────────────────────────────┤
    │ Conv Block 2                │
    │ Conv2d(32→64, 3x3) + BN +  │
    │ ReLU + MaxPool2d(2) +       │
    │ Dropout2d(p2)               │
    ├─────────────────────────────┤
    │ Conv Block 3                │
    │ Conv2d(64→128, 3x3) + BN + │
    │ ReLU + MaxPool2d(2) +       │
    │ Dropout2d(p2)               │
    ├─────────────────────────────┤
    │ Conv Block 4 (финальный)    │
    │ Conv2d(128→256, 3x3) + BN +│
    │ ReLU + AdaptiveAvgPool(1,1) │
    ├─────────────────────────────┤
    │ Classifier                  │
    │ Flatten → Linear(256→128) → │
    │ ReLU → Dropout(p3) →        │
    │ Linear(128→20)              │
    └─────────────────────────────┘
          │
          ▼
    Выход: (batch, 20) — логиты
```

- Все свёртки `3x3`, padding=1, bias=False
- BatchNorm после каждой свёртки
- Количество свёрточных блоков: 3 или 4 (подбирается Optuna)
- Каналы: `1 → 32 → 64 → 128 → 256`

## 5. Процесс обучения

**Модуль:** `ml/train.py`

### Этап 1 — Подбор гиперпараметров (Optuna)

- **Количество триалов:** 40
- **Sampler:** TPE (Tree-structured Parzen Estimator)
- **Pruner:** MedianPruner (отсечение неперспективных триалов)
- **Метрика:** максимизация accuracy на валидации

**Пространство поиска:**

| Параметр         | Диапазон            |
|:-----------------|:--------------------|
| n_mels           | {32, 64, 128}       |
| batch_size       | {16, 32, 64}        |
| learning_rate    | [1e-4, 5e-3] (log)  |
| weight_decay     | [1e-5, 1e-3] (log)  |
| dropout_p1       | [0.1, 0.4]          |
| dropout_p2       | [0.1, 0.4]          |
| dropout_p3       | [0.3, 0.6]          |
| optimizer        | {Adam, AdamW}       |
| num_conv_blocks  | {3, 4}              |
| epochs           | [30, 100]           |

### Этап 2 — Финальное обучение

С лучшими параметрами из Optuna:

- **Loss:** CrossEntropyLoss
- **Scheduler:** ReduceLROnPlateau (patience=7, factor=0.5)
- **Gradient clipping:** max_norm=1.0
- **Early stopping:** patience=15 эпох без улучшения
- **Устройство:** MPS (Apple Silicon GPU) или CPU

### Логирование

Каждые 5 эпох выводится:
```
Epoch  10/78 | tr 2.1234/0.3500 | val 2.3456/0.2800 | lr=1.00e-03 | 2.1s ★
```
- `tr loss/acc` — потери и точность на обучении
- `val loss/acc` — потери и точность на валидации
- `lr` — текущий learning rate
- `★` — лучший результат на валидации

## 6. Сохранение модели

После обучения сохраняются три файла:

| Файл                          | Содержимое                                      |
|:------------------------------|:------------------------------------------------|
| `saved_models/model.pth`      | Веса модели (`state_dict`)                      |
| `saved_models/model_checkpoint.pth` | Полный чекпоинт: веса + label_map + best_params + val_acc |
| `saved_models/model.h5`       | Веса в формате HDF5 + метаданные в атрибутах    |
| `saved_models/label_map.json` | Карта меток: название планеты → целое число      |
| `training_logs/history.json`  | История обучения: loss/acc по эпохам + параметры |

### Формат `model_checkpoint.pth`:
```python
{
    "state_dict": model.state_dict(),
    "label_map": {"55_Cancri_Bc": 0, ...},
    "best_params": {"n_mels": 64, "lr": 0.001, ...},
    "val_acc": 0.85,
    "num_classes": 20,
}
```

### Формат `history.json`:
```json
{
    "best_params": {...},
    "best_val_acc": 0.85,
    "num_classes": 20,
    "label_map": {...},
    "history": {
        "train_loss": [3.0, 2.8, ...],
        "train_acc": [0.05, 0.10, ...],
        "val_loss": [3.0, 2.9, ...],
        "val_acc": [0.05, 0.08, ...]
    }
}
```

## 7. Загрузка модели для инференса

**Модуль:** `app/services/ml_service.py`

```python
checkpoint = torch.load("saved_models/model_checkpoint.pth")
model = build_model(
    num_classes=checkpoint["num_classes"],
    n_mels=checkpoint["best_params"]["n_mels"],
    ...
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

## 8. Запуск обучения

```bash
cd /path/to/predprof
python -m ml.train
```

Входные данные: `Data/Data(1).npz` (архив с train_x, train_y, valid_x, valid_y).

## 9. Структура входных данных

**Файл:** `Data(1).npz`

| Массив    | Размер          | Описание                         |
|:----------|:----------------|:---------------------------------|
| train_x   | (1200, 80000, 1) | Обучающие wav-сигналы            |
| train_y   | (1200,)          | Повреждённые метки (строки)      |
| valid_x   | (400, 80000, 1)  | Валидационные wav-сигналы        |
| valid_y   | (400,)           | Повреждённые метки (строки)      |
| vaild_y   | (1200,)          | Дубль (опечатка в имени ключа)   |

## 10. Зависимости для обучения

```
torch >= 2.0
torchaudio
numpy
optuna
h5py
scikit-learn
```
