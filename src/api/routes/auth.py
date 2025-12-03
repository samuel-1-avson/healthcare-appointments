"""
Authentication Routes
=====================
Endpoints for login and user management.
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..auth import (
    Token, User, authenticate_user, create_access_token, get_current_user
)
from ..config import get_settings
from ..database import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db)
):
    """
    Login to get an access token.
    
    Requires username and password.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    settings = get_settings()
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get current logged-in user details.
    
    Protected endpoint.
    """
    return current_user


@router.get("/login/oidc")
async def login_oidc(redirect_uri: str = "http://localhost:8000/api/v1/auth/callback/oidc"):
    """
    Initiate OIDC login flow.
    """
    from ..security.oidc import get_oidc_client
    from fastapi.responses import RedirectResponse
    
    oidc = get_oidc_client()
    try:
        url = await oidc.get_login_url(redirect_uri)
        return RedirectResponse(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/callback/oidc")
async def callback_oidc(
    code: str,
    redirect_uri: str = "http://localhost:8000/api/v1/auth/callback/oidc",
    db: Session = Depends(get_db)
):
    """
    Handle OIDC callback.
    """
    from ..security.oidc import get_oidc_client
    from ..models import User as UserModel
    from ..auth import create_access_token
    
    oidc = get_oidc_client()
    try:
        # Exchange code for tokens
        tokens = await oidc.exchange_code(code, redirect_uri)
        id_token = tokens.get("id_token")
        
        # Verify ID token
        claims = await oidc.verify_id_token(id_token)
        email = claims.get("email")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in token")
            
        # Find or create user
        user = db.query(UserModel).filter(UserModel.email == email).first()
        if not user:
            # Create new user (JIT provisioning)
            user = UserModel(
                username=email.split("@")[0],
                email=email,
                full_name=claims.get("name"),
                role="user",
                hashed_password="oidc_user"  # Placeholder, cannot login with password
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
        # Create access token
        settings = get_settings()
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")
