"""
Authentication Utilities
========================
Handles password hashing, token generation, and user authentication.
"""

from datetime import datetime, timedelta
from typing import Optional, Union, Annotated

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import get_settings
from .database import get_db
from .models import User as UserModel

# Password hashing context
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


# --- Schemas ---

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None
    role: str = "user"

    class Config:
        from_attributes = True


class UserInDB(User):
    hashed_password: str


# --- Core Functions ---

def get_user(db: Session, username: str) -> Optional[UserModel]:
    """Get user from database."""
    return db.query(UserModel).filter(UserModel.username == username).first()


def authenticate_user(db: Session, username: str, password: str) -> Union[UserModel, bool]:
    """Authenticate user credentials."""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Parameters
    ----------
    data : dict
        Payload data
    expires_delta : timedelta, optional
        Expiration time
    
    Returns
    -------
    str
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    
    settings = get_settings()
    # Use a default secret for dev if not set (INSECURE for prod, but handled in config)
    secret_key = getattr(settings, "secret_key", "dev_secret_key_change_me")
    algorithm = getattr(settings, "algorithm", "HS256")
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    Validate token and get current user.
    
    Used as a dependency for protected routes.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    settings = get_settings()
    
    # Check for debug mode bypass
    if token is None:
        if settings.debug:
            return User(
                username="dev_user",
                email="dev@example.com",
                full_name="Developer User",
                disabled=False,
                role="admin"
            )
        else:
            raise credentials_exception

    secret_key = getattr(settings, "secret_key", "dev_secret_key_change_me")
    algorithm = getattr(settings, "algorithm", "HS256")
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    # Fetch user from DB
    user = get_user(db, username=token_data.username)
    
    if user is None:
        raise credentials_exception
        
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        role=user.role
    )


async def get_current_admin_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Validate that the current user has admin privileges.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Validate that the current user is active.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
