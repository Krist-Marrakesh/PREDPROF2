пр# Alien Radio Signal Classifier

Программный продукт для классификации радиосигналов инопланетных цивилизаций с помощью обученной нейронной сети. Веб-приложение с авторизацией, загрузкой данных и аналитикой.

## Оглавление

- [Структура проекта](#структура-проекта)
- [Установка и запуск](#установка-и-запуск)
- [Архитектура](#архитектура)
- [Восстановление меток](#восстановление-меток)
- [Обучение модели](#обучение-модели)
- [Веб-приложение](#веб-приложение)
- [API](#api)
- [Тестирование](#тестирование)
- [Стек технологий](#стек-технологий)

---

## Структура проекта

```
predprof/
├── app/                        # Веб-приложение (FastAPI)
│   ├── main.py                 # Инициализация приложения, подключение роутеров
│   ├── config.py               # Конфигурация: пути, секреты, БД
│   ├── database.py             # Подключение к SQLite через SQLAlchemy
│   ├── models.py               # ORM-модель User
│   ├── schemas.py              # Pydantic-схемы запросов/ответов
│   ├── routes/
│   │   ├── auth.py             # Авторизация: /login, /logout
│   │   ├── admin.py            # Админ-панель: /admin, /admin/create-user
│   │   └── user.py             # Пользователь: /dashboard, /upload, /analytics
│   ├── services/
│   │   ├── auth_service.py     # Хеширование паролей, JWT-токены, проверка ролей
│   │   ├── ml_service.py       # Загрузка модели, предсказания, история обучения
│   │   └── data_service.py     # Парсинг загруженных файлов (.npy/.npz)
│   ├── templates/              # Jinja2 HTML-шаблоны
│   │   ├── base.html           # Базовый макет
│   │   ├── login.html          # Форма входа
│   │   ├── dashboard.html      # Панель пользователя + загрузка данных
│   │   ├── admin.html          # Админ-панель + создание пользователей
│   │   └── analytics.html      # Графики и диаграммы
│   └── static/
│       ├── css/style.css       # Стили
│       └── js/charts.js        # Построение графиков (Chart.js)
│
├── ml/                         # ML-модуль
│   ├── model.py                # AudioClassifier — архитектура CNN
│   ├── train.py                # Обучение: Optuna + финальное обучение
│   ├── preprocess.py           # Предобработка: wav -> Mel-спектрограмма
│   └── dataset.py              # PyTorch Dataset с аугментацией
│
├── tests/                      # Unit-тесты (pytest)
│   ├── test_auth.py            # Тесты авторизации (11 тестов)
│   ├── test_routes.py          # Тесты маршрутов и админ-панели (5 тестов)
│   ├── test_model.py           # Тесты архитектуры модели (7 тестов)
│   └── test_data.py            # Тесты предобработки и датасета (8 тестов)
│
├── Data/                       # Набор данных
│   └── Data(1).npz             # Архив: train_x, train_y, valid_x, valid_y
│
├── saved_models/               # Сохраненные модели
│   ├── model.pth               # Веса лучшей модели
│   ├── model_checkpoint.pth    # Полный чекпоинт (веса + параметры + label_map)
│   ├── model.h5                # Экспорт весов в HDF5
│   └── label_map.json          # Карта: название цивилизации -> целое число
│
├── training_logs/              # Логи обучения
│   └── history.json            # История: loss/acc по эпохам + лучшие параметры
│
├── notebooks/                  # Jupyter-ноутбуки
│   ├── eda.ipynb               # Исследовательский анализ данных
│   └── train.ipynb             # Обучение в ноутбуке
│
├── run.py                      # Точка входа (uvicorn)
├── requirements.txt            # Зависимости Python
├── TRAINING.md                 # Документация по обучению
└── predprof.db                 # SQLite база данных (runtime)
```

---

## Установка и запуск

### 1. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
python -m ml.train
```

Загружает `Data/Data(1).npz`, восстанавливает метки, подбирает гиперпараметры через Optuna (40 триалов), обучает финальную модель. Сохраняет результаты в `saved_models/` и `training_logs/`.

### 3. Запуск веб-приложения

```bash
python run.py
```

Приложение доступно по адресу `http://localhost:8000`.

### 4. Вход по умолчанию

- **Администратор:** логин `admin`, пароль `admin`

### 5. Запуск тестов

```bash
pytest tests/ -v
```

---

## Архитектура

### Общая схема

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Browser   │────>│   FastAPI    │────>│   SQLite DB  │
│  (Jinja2)   │<────│   (Routes)   │<────│  (Users)     │
└─────────────┘     └──────┬───────┘     └──────────────┘
                           │
                    ┌──────┴───────┐
                    │   Services   │
                    ├──────────────┤
                    │ auth_service │  JWT + bcrypt
                    │ ml_service   │  PyTorch inference
                    │ data_service │  .npy/.npz parsing
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   ML Module  │
                    ├──────────────┤
                    │ model.py     │  AudioClassifier (CNN)
                    │ preprocess.py│  Wav -> Mel spectrogram
                    │ train.py     │  Optuna + training loop
                    │ dataset.py   │  PyTorch Dataset
                    └──────────────┘
```

### Модуль `app/` — Веб-приложение

| Файл | Назначение |
|:-----|:-----------|
| `main.py` | Создание FastAPI-приложения, подключение роутеров, создание таблиц в БД, seed администратора |
| `config.py` | Все конфигурационные константы: SECRET_KEY, пути к моделям, БД |
| `database.py` | SQLAlchemy engine + session factory для SQLite |
| `models.py` | ORM-модель `User` (id, username, password_hash, first_name, last_name, role, created_at) |
| `schemas.py` | Pydantic-модели: `UserCreate`, `UserOut`, `LoginForm` |

### Модуль `app/routes/` — Маршруты

| Маршрут | Метод | Роль | Описание |
|:--------|:------|:-----|:---------|
| `/` | GET | - | Редирект на `/login` |
| `/login` | GET | - | Страница входа |
| `/login` | POST | - | Аутентификация, установка JWT-cookie |
| `/logout` | GET | - | Удаление cookie, выход |
| `/admin` | GET | admin | Список пользователей |
| `/admin/create-user` | POST | admin | Создание нового пользователя |
| `/dashboard` | GET | user | Панель загрузки данных + результаты предсказаний |
| `/upload` | POST | user | Загрузка .npy/.npz файла, получение предсказаний |
| `/analytics` | GET | user | Страница с графиками |
| `/api/analytics/data` | GET | user | JSON с данными для графиков |

### Модуль `app/services/` — Бизнес-логика

**auth_service.py:**
- `hash_password(password)` — bcrypt-хеширование
- `verify_password(plain, hashed)` — проверка пароля
- `create_token(username, role)` — JWT-токен (HS256, срок 8 часов)
- `get_current_user(request, db)` — извлечение пользователя из cookie
- `require_admin(request, db)` — проверка роли администратора

**ml_service.py:**
- `is_model_ready()` — ленивая загрузка модели из чекпоинта
- `predict_batch(X)` — пакетное предсказание: возвращает класс, уверенность, top-3
- `get_training_history()` — чтение `history.json`
- `get_model_info()` — информация о модели (точность, количество классов)

**data_service.py:**
- `parse_upload(content, filename)` — парсинг .npy/.npz в numpy-массив

### Модуль `ml/` — Машинное обучение

**model.py — AudioClassifier (CNN):**
```
Вход: (batch, 1, n_mels, time_steps)
  -> Conv2d(1->32) + BN + ReLU + MaxPool2d + Dropout
  -> Conv2d(32->64) + BN + ReLU + MaxPool2d + Dropout
  -> Conv2d(64->128) + BN + ReLU + MaxPool2d + Dropout
  -> Conv2d(128->256) + BN + ReLU + AdaptiveAvgPool2d(1,1)
  -> Flatten -> Linear(256->128) -> ReLU -> Dropout -> Linear(128->20)
Выход: (batch, 20) — логиты по 20 классам
```

**preprocess.py — Пайплайн предобработки:**
1. Извлечение активной области (RMS-порог)
2. Pad/truncate до 80000 отсчетов
3. MelSpectrogram (sr=22050, n_fft=1024, hop=512, n_mels=32/64/128)
4. AmplitudeToDB (power, top_db=80)
5. Z-нормализация

**train.py — Обучение:**
1. Загрузка `Data(1).npz`, восстановление меток
2. Предвычисление спектрограмм для n_mels = {32, 64, 128}
3. Optuna: 40 триалов, TPE sampler, MedianPruner
4. Финальное обучение с лучшими параметрами + early stopping
5. Сохранение model.pth, model_checkpoint.pth, model.h5, history.json

---

## Восстановление меток

Исходные метки повреждены практикантами. Вместо целых чисел содержат строки вида:
```
0256aa0ac353680c35899c806079337fGliese_163_c
```

**Алгоритм восстановления:**
1. Регулярное выражение `^[0-9a-f]{32}` удаляет 32-символьный hex-префикс
2. Извлекается название цивилизации (планеты)
3. Уникальные названия сортируются по алфавиту
4. Каждому присваивается целое число от 0 до 19

**Реализация:** `ml/preprocess.py:extract_label()`

**Результат — 20 классов:**

| ID | Цивилизация | ID | Цивилизация |
|---:|:------------|---:|:------------|
| 0 | 55_Cancri_Bc | 10 | K2-72e |
| 1 | Gliese_ | 11 | Kepler-155c |
| 2 | Gliese_12_b | 12 | Kepler-174d |
| 3 | Gliese_163_c | 13 | Kepler-186f |
| 4 | HD_20794_d | 14 | Kepler-22b |
| 5 | HD_216520_c | 15 | Kepler-283c |
| 6 | HIP_38594_b | 16 | Kepler-296e |
| 7 | K2-155d | 17 | Kepler-296f |
| 8 | K2-288Bb | 18 | Kepler-62e |
| 9 | K2-332b | 19 | Kepler-62f |

---

## Обучение модели

Подробная документация: [TRAINING.md](TRAINING.md)

### Краткий обзор

```bash
python -m ml.train
```

1. **Данные:** 1200 обучающих + 400 валидационных записей (wav, 80000 отсчетов)
2. **Предобработка:** wav -> Mel-спектрограмма -> z-нормализация
3. **Подбор гиперпараметров:** Optuna, 40 триалов (n_mels, lr, dropout, batch_size и др.)
4. **Обучение:** CrossEntropyLoss, Adam/AdamW, ReduceLROnPlateau, gradient clipping, early stopping
5. **Аугментация:** гауссов шум, time/frequency masking
6. **Устройство:** MPS (Apple Silicon GPU) с фолбэком на CPU

### Сохранение модели

| Файл | Содержимое |
|:-----|:-----------|
| `saved_models/model.pth` | Веса модели (state_dict) |
| `saved_models/model_checkpoint.pth` | Полный чекпоинт: веса + label_map + параметры + val_acc |
| `saved_models/model.h5` | HDF5-экспорт весов + метаданные в атрибутах |
| `saved_models/label_map.json` | Карта: название планеты -> целое число |
| `training_logs/history.json` | История: train_loss, train_acc, val_loss, val_acc по эпохам |

---

## Веб-приложение

### Роли пользователей

**Администратор:**
- Просмотр списка всех пользователей
- Создание новых пользователей (Имя, Фамилия, логин, пароль, роль)

**Пользователь:**
- Загрузка набора данных для проверки модели (.npy/.npz)
- Просмотр результатов классификации (класс, уверенность, top-3)
- Просмотр аналитики

### Аналитика

Страница `/analytics` отображает:
1. **График точности на валидации** от количества эпох обучения
2. **Диаграмма распределения классов** — количество записей каждой цивилизации в обучающей выборке
3. **Диаграмма точности** определения каждой записи из тестового набора данных
4. **Top-5 классов** — наиболее часто встречающиеся классы в валидационной выборке

Все графики поддерживают масштабирование.

### Авторизация

- Пароли хранятся в виде bcrypt-хешей в SQLite
- Сессия — JWT-токен в httponly-cookie (срок 8 часов, алгоритм HS256)
- Роль (admin/user) определяет доступные маршруты

### База данных

**СУБД:** SQLite (`predprof.db`)

**Таблица `users`:**

| Поле | Тип | Описание |
|:-----|:----|:---------|
| id | INTEGER PK | Автоинкремент |
| username | VARCHAR(64) UNIQUE | Логин |
| password_hash | TEXT | bcrypt-хеш пароля |
| first_name | VARCHAR(64) | Имя |
| last_name | VARCHAR(64) | Фамилия |
| role | VARCHAR(16) | "admin" или "user" |
| created_at | DATETIME | Дата создания |

---

## API

### Эндпоинты

| Endpoint | Метод | Описание | Авторизация |
|:---------|:------|:---------|:------------|
| `GET /` | GET | Редирект на /login | - |
| `GET /login` | GET | HTML-форма входа | - |
| `POST /login` | POST | Вход (Form: username, password) | - |
| `GET /logout` | GET | Выход | - |
| `GET /admin` | GET | Админ-панель (HTML) | admin |
| `POST /admin/create-user` | POST | Создание пользователя (Form) | admin |
| `GET /dashboard` | GET | Панель пользователя (HTML) | user |
| `POST /upload` | POST | Загрузка файла (multipart) | user |
| `GET /analytics` | GET | Страница аналитики (HTML) | user |
| `GET /api/analytics/data` | GET | Данные для графиков (JSON) | user |

### Формат ответа `/api/analytics/data`

```json
{
  "ready": true,
  "best_val_acc": 0.85,
  "epochs": [1, 2, 3, ...],
  "train_acc": [0.05, 0.10, ...],
  "val_acc": [0.05, 0.08, ...],
  "train_loss": [3.0, 2.8, ...],
  "val_loss": [3.0, 2.9, ...],
  "train_class_counts": {"55_Cancri_Bc": 76, "Gliese_": 62, ...},
  "top5_valid": {
    "classes": ["Kepler-62f", "Gliese_12_b", ...],
    "counts": [34, 23, ...]
  },
  "predictions": [
    {"index": 0, "predicted_class": "Kepler-22b", "confidence": 0.87, "top3": [...]}
  ]
}
```

### Формат предсказания

```json
{
  "index": 0,
  "predicted_class": "Kepler-22b",
  "confidence": 0.87,
  "top3": [
    {"class": "Kepler-22b", "prob": 0.87},
    {"class": "Kepler-62e", "prob": 0.05},
    {"class": "Gliese_163_c", "prob": 0.03}
  ]
}
```

---

## Тестирование

```bash
pytest tests/ -v
```

### Тесты авторизации (`test_auth.py`)
- Редирект `/` -> `/login`
- Страница входа возвращает 200
- Неверные учетные данные -> 401
- Админ -> редирект на `/admin`
- Пользователь -> редирект на `/dashboard`
- Выход очищает cookie
- Админ-панель доступна для админа
- Админ-панель запрещена для пользователя (403)
- Dashboard требует авторизацию (401)
- Dashboard доступен авторизованному пользователю

### Тесты маршрутов (`test_routes.py`)
- Создание пользователя администратором
- Ошибка при создании дубликата пользователя
- API аналитики возвращает JSON
- API аналитики требует авторизацию
- Страница аналитики возвращает HTML

### Тесты модели (`test_model.py`)
- Выходная размерность при 4 блоках
- Выходная размерность при 3 блоках
- Работа с разными n_mels (32, 64, 128)
- AdaptiveAvgPool2d в последнем блоке
- Отсутствие MaxPool2d в последнем блоке
- Структура классификатора
- Работа с batch_size=1

### Тесты данных (`test_data.py`)
- Удаление hex-префикса из метки
- Passthrough для чистых меток
- Построение отсортированной карты меток
- Корректное кодирование меток
- Форма мел-спектрограммы
- Padding короткого сигнала
- Truncate длинного сигнала
- Нормализация мел-спектрограммы
- Форма элемента датасета

---

## Стек технологий

| Компонент | Технология | Версия |
|:----------|:-----------|:-------|
| Backend | FastAPI | 0.135.1 |
| Сервер | Uvicorn | - |
| ORM | SQLAlchemy | 2.0.48 |
| Шаблоны | Jinja2 | 3.1.6 |
| Валидация | Pydantic | 2.12.5 |
| Хеширование | bcrypt | 5.0.0 |
| JWT | PyJWT | 2.12.1 |
| ML Framework | PyTorch | 2.10.0 |
| Аудио | torchaudio | 2.10.0 |
| HPO | Optuna | 4.7.0 |
| Данные | NumPy | 2.4.3 |
| СУБД | SQLite | встроенная |
| Тесты | pytest | - |

---

## Регламент демонстрации

1. Демонстрация лога обучения (`training_logs/history.json`) и методов сохранения модели (`model.pth`, `model_checkpoint.pth`, `model.h5`)
2. Запуск: `python run.py` -> `http://localhost:8000`
3. Авторизация администратора: `admin` / `admin`
4. Создание пользователя по данным жюри
5. Авторизация пользователем
6. Загрузка тестового набора данных через `/dashboard`
7. Просмотр точности и потерь на тестовом наборе
8. Просмотр графиков и диаграмм на `/analytics`
9. Демонстрация `predprof.db` (персистентное хранение пользователей)
10. Запуск тестов: `pytest tests/ -v`
