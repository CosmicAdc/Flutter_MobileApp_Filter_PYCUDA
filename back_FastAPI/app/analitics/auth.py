from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .databases import SessionLocal, engine
from . import schemas, crud, user as user_model


user_model.Base.metadata.create_all(bind=engine)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register/", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Correo ya registrado")
    return crud.create_user(db=db, user=user)

@router.post("/login/")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(user_model.User).filter(user_model.User.email == user.email, user_model.User.password == user.password).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="Credenciales invalidas no existe")
    return {"message": "Login correcto", "user": db_user.username, "id": db_user.id}


##Creación de post
@router.post("/posts/")
def create_post(post: schemas.PostCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, id_user=post.id_user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.create_post(db=db, post=post)


# Obtener todos los post
@router.get("/posts/", response_model=list[schemas.PostOut])
def read_all_posts(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    posts = crud.get_all_posts(db, skip=skip, limit=limit)
    posts_with_username = []
    for post in posts:
        post_dict = post.__dict__
        user_id = post_dict['id_user']
        user = crud.get_user(db, user_id)  
        post_dict['username'] = user.username  # Agregar el nombre de usuario al diccionario del post
        posts_with_username.append(post_dict)
    return posts_with_username



# Obtener post por Usuario
@router.get("/users/{id_user}/posts", response_model=list[schemas.PostOut])
def read_posts_by_user(id_user: int, skip: int = 0, limit: int = 30, db: Session = Depends(get_db)):
    db_user = crud.get_users(db, id_user=id_user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.get_posts_by_user(db=db, id_user=id_user, skip=skip, limit=limit)