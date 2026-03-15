import json

from fastapi import APIRouter, Request, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.auth_service import get_current_user
from app.services.ml_service import predict_batch, get_training_history, get_model_info, is_model_ready
from app.services.data_service import parse_upload

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

_last_predictions: list = []


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    model_info = get_model_info()
    return templates.TemplateResponse(request, "dashboard.html", {"current_user": user,
        "model_info": model_info,
        "predictions": _last_predictions,
        "error": None,
    })


@router.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    global _last_predictions
    user = get_current_user(request, db)
    model_info = get_model_info()

    if not is_model_ready():
        return templates.TemplateResponse(request, "dashboard.html", {"current_user": user,
            "model_info": model_info, "predictions": [],
            "error": "Модель ещё не обучена. Запустите ml/train.py",
        })

    try:
        content = await file.read()
        X = parse_upload(content, file.filename)
        _last_predictions = predict_batch(X)
    except Exception as e:
        return templates.TemplateResponse(request, "dashboard.html", {"current_user": user,
            "model_info": model_info, "predictions": [],
            "error": f"Ошибка обработки файла: {e}",
        })

    return templates.TemplateResponse(request, "dashboard.html", {"current_user": user,
        "model_info": model_info, "predictions": _last_predictions,
        "error": None,
    })


@router.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    return templates.TemplateResponse(request, "analytics.html", {"current_user": user,
    })


@router.get("/api/analytics/data")
async def analytics_data(request: Request, db: Session = Depends(get_db)):
    get_current_user(request, db)
    history = get_training_history()
    if not history:
        return {"ready": False}

    h = history["history"]
    epochs = list(range(1, len(h["train_acc"]) + 1))
    label_map = history.get("label_map", {})

    train_class_counts = {}
    try:
        import numpy as np
        from app.config import DATA_DIR
        train_y = np.load(DATA_DIR / "train_y.npy", allow_pickle=True)
        from ml.preprocess import extract_label
        for lbl in train_y:
            name = extract_label(lbl)
            train_class_counts[name] = train_class_counts.get(name, 0) + 1
    except Exception:
        pass

    valid_class_counts = {}
    try:
        valid_y = np.load(DATA_DIR / "valid_y.npy", allow_pickle=True)
        for lbl in valid_y:
            name = extract_label(lbl)
            valid_class_counts[name] = valid_class_counts.get(name, 0) + 1
    except Exception:
        pass

    top5_valid = sorted(valid_class_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "ready": True,
        "best_val_acc": history["best_val_acc"],
        "epochs": epochs,
        "train_acc": h["train_acc"],
        "val_acc": h["val_acc"],
        "train_loss": h["train_loss"],
        "val_loss": h["val_loss"],
        "train_class_counts": train_class_counts,
        "top5_valid": {"classes": [x[0] for x in top5_valid], "counts": [x[1] for x in top5_valid]},
        "predictions": _last_predictions,
    }
