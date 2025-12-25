from datetime import datetime, timedelta
from typing import Optional
import secrets
import hashlib
import json
from fastapi import HTTPException
from pydantic import BaseModel
import os
import jwt
from jwt import PyJWTError
from .database_models import User, engine, SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Secret key for JWT signing (use environment variable in production)
SECRET_KEY = os.getenv("AUTH_SECRET", "fallback-secret-for-development-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class UserSchema(BaseModel):
    id: int
    email: str
    name: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class TokenData(BaseModel):
    email: str

def hash_password(password: str) -> str:
    """Hash a password"""
    # Truncate password to 72 bytes to comply with bcrypt limitations
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        # Truncate at byte level and decode to ensure final byte length <= 72
        password = password_bytes[:72].decode('utf-8', errors='ignore')
        # Double-check the byte length after decoding to ensure compliance
        if len(password.encode('utf-8')) > 72:
            # If decoded string is still > 72 bytes, use a more robust truncation
            # that ensures the final byte length is definitely <= 72
            # Start with a safe character count and reduce if needed
            for i in range(min(len(password), 72), 0, -1):
                candidate = password[:i]
                if len(candidate.encode('utf-8')) <= 72:
                    password = candidate
                    break
            else:
                # Ultimate fallback: use a conservative length that's guaranteed safe
                password = password[:60] if len(password) > 60 else password

    # Final safety check to ensure byte length <= 72 before hashing
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        # If somehow still over 72 bytes, do final byte-level truncation
        password = password_bytes[:72].decode('utf-8', errors='ignore')

    try:
        return pwd_context.hash(password)
    except Exception as e:
        # If bcrypt fails for any reason (including version compatibility issues),
        # log the error and use a fallback approach
        print(f"bcrypt error: {str(e)}")
        # Ensure password is within limits and use a fallback method
        # This is a safety fallback in case bcrypt library has issues
        safe_password = password.encode('utf-8')[:70].decode('utf-8', errors='ignore')
        return pwd_context.hash(safe_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticate_user(db: Session, email: str, password: str):
    """Authenticate a user by email and password"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email"""
    return db.query(User).filter(User.email == email).first()

def register_user(db: Session, email: str, password: str, name: str) -> User:
    """Register a new user"""
    # Check if user already exists
    existing_user = get_user_by_email(db, email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pwd = hash_password(password)

    user = User(
        email=email,
        name=name,
        hashed_password=hashed_pwd
    )

    db.add(user)
    try:
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered")

def get_current_user_from_token(token: str) -> Optional[User]:
    """Get the current user from the token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None

        # Create a new session to get the user from database
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            return user
        finally:
            db.close()
    except jwt.PyJWTError:
        return None


def get_current_user(token: str) -> Optional[User]:
    """Alias for get_current_user_from_token for compatibility"""
    return get_current_user_from_token(token)

# OAuth state storage (in production, use a more secure method)
oauth_states = {}

def generate_oauth_state() -> str:
    """Generate a state parameter for OAuth flow"""
    state = secrets.token_urlsafe(32)
    oauth_states[state] = datetime.utcnow() + timedelta(minutes=5)  # 5 min expiry
    return state

def validate_oauth_state(state: str) -> bool:
    """Validate an OAuth state parameter"""
    if state not in oauth_states:
        return False

    expiry = oauth_states[state]
    if datetime.utcnow() > expiry:
        del oauth_states[state]
        return False

    del oauth_states[state]
    return True