from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.services.auth_service import require_admin, hash_password

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, db: Session = Depends(get_db)):
    admin = require_admin(request, db)
    users = db.query(User).order_by(User.created_at.desc()).all()
    return templates.TemplateResponse(request, "admin.html", {"current_user": admin, "users": users, "error": None, "success": None,
    })


@router.post("/admin/create-user")
async def create_user(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    role: str = Form("user"),
    db: Session = Depends(get_db),
):
    admin = require_admin(request, db)
    users = db.query(User).order_by(User.created_at.desc()).all()

    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse(request, "admin.html", {"current_user": admin, "users": users,
            "error": f"Пользователь '{username}' уже существует", "success": None,
        })

    new_user = User(
        username=username,
        password_hash=hash_password(password),
        first_name=first_name,
        last_name=last_name,
        role=role,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    users = db.query(User).order_by(User.created_at.desc()).all()
    return templates.TemplateResponse(request, "admin.html", {"current_user": admin, "users": users,
        "error": None, "success": f"Пользователь '{username}' создан",
    })
