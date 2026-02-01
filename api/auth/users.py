from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password): return pwd_context.hash(password)
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)



from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models import User
from .password_utils import hash_password

router = APIRouter()

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@router.post("/register")
def register(email: str, password: str, phone: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered")
    user = User(email=email, password=hash_password(password), phone=phone)
    db.add(user); db.commit()
    return {"status": "success", "email": email}
