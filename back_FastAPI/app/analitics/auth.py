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
    return {"message": "Login correcto", "user": db_user.username}
