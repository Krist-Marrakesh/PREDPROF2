from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.database import engine, SessionLocal
from app.models import Base, User
from app.services.auth_service import hash_password
from app.routes import auth, admin, user

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Alien Radio Signal Classifier")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(user.router)


def _seed_admin():
    db: Session = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            db.add(User(
                username="admin",
                password_hash=hash_password("admin"),
                first_name="Admin",
                last_name="System",
                role="admin",
            ))
            db.commit()
    finally:
        db.close()


_seed_admin()
