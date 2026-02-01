import jwt
from datetime import datetime, timedelta

SECRET_KEY = "PAYWATCH_SUPER_SECRET"
ALGORITHM = "HS256"

def create_token(email, role):
    payload = {
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=4)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token):
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
