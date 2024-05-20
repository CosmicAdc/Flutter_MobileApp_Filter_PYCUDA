from sqlalchemy import Column, Integer, String
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from .databases import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)


class Posts(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    id_user = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    image_path = Column(String(255), unique=True, index=True)
    description = Column(Text)