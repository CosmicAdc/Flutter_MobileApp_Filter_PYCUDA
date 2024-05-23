from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        orm_mode = True

class PostCreate(BaseModel):
    id_user: int
    image_path: str
    description: str

class PostOut(BaseModel):
    id: int
    id_user: int
    username: Optional[str]
    image_path: str
    description: str

    class Config:
        orm_mode = True


