from sqlalchemy.orm import Session

from . import schemas, user as user_model


def get_user_by_email(db: Session, email: str):
    return db.query(user_model.User).filter(user_model.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate):
    db_user = user_model.User(username=user.username, email=user.email, password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user