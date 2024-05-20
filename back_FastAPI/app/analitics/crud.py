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


def get_user(db: Session, id_user: int):
    return db.query(user_model.User).filter(user_model.User.id == id_user).first()

def create_post(db: Session, post: schemas.PostCreate):
    db_post = user_model.Posts(id_user=post.id_user, image_path=post.image_path, description=post.description)
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return db_post

def get_post(db: Session, post_id: int):
    return db.query(user_model.Posts).filter(user_model.Posts.id == post_id).first()

def get_posts(db: Session, skip: int = 0, limit: int = 10):
    return db.query(user_model.Posts).offset(skip).limit(limit).all()

def get_posts_by_user(db: Session, id_user: int, skip: int = 0, limit: int = 10):
    return db.query(user_model.Posts).filter(user_model.Posts.id_user == id_user).offset(skip).limit(limit).all()

def get_all_posts(db: Session, skip: int = 0, limit: int = 10):
    return db.query(user_model.Posts).offset(skip).limit(limit).all()