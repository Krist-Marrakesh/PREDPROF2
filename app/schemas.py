from datetime import datetime
from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    password: str
    first_name: str
    last_name: str
    role: str = "user"


class UserOut(BaseModel):
    id: int
    username: str
    first_name: str
    last_name: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


class LoginForm(BaseModel):
    username: str
    password: str
