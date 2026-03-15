import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.models import User
from app.services.auth_service import hash_password


@pytest.fixture(autouse=True)
def setup_db(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    TestSession = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = TestSession()
    db.add(User(username="testadmin", password_hash=hash_password("pass123"),
                first_name="Test", last_name="Admin", role="admin"))
    db.add(User(username="testuser", password_hash=hash_password("user123"),
                first_name="Test", last_name="User", role="user"))
    db.commit()
    db.close()

    def override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app, follow_redirects=False) as c:
        yield c


def login_cookie(c, username, password):
    r = c.post("/login", data={"username": username, "password": password})
    return r.headers["set-cookie"].split(";")[0]
