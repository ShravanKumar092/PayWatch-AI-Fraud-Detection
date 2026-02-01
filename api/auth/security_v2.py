from datetime import datetime, timedelta
from jose import jwt, JWTError
import bcrypt
import hashlib

SECRET_KEY = "paywatch-super-secret"  # CHANGE FOR DEPLOYMENT
ALGORITHM = "HS256"
ACCESS_EXPIRE_MINUTES = 60


def hash_password(password: str):
    """
    Hash password using bcrypt with SHA256 pre-hashing.
    Always supports arbitrary length passwords.
    """
    if not password:
        raise ValueError("Password cannot be empty")

    try:
        password_bytes = password.encode("utf-8")
    except (UnicodeEncodeError, AttributeError) as e:
        raise ValueError(f"Invalid password encoding: {str(e)}")

    # Pre-hash with SHA256 to stay under bcrypt's 72-byte limit
    sha256_hash = hashlib.sha256(password_bytes).hexdigest()
    sha256_bytes = sha256_hash.encode("utf-8")  # 64 bytes

    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(sha256_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str):
    if not plain_password or not hashed_password:
        return False
    try:
        sha256_hash = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
        return bcrypt.checkpw(sha256_hash.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded
    except JWTError:
        return None

