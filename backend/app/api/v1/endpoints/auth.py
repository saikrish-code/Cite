"""Authentication REST API endpoints for user signup, login, and profile access."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
)
from app.models.user import User
from app.schemas.user import TokenResponse, UserLogin, UserRegister, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
    description="Create a new user account with email and password, returning an access token.",
)
async def signup(
    payload: UserRegister,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Create a new user account with bcrypt-hashed credentials.

    Args:
        payload: Registration details containing email, password, and optional full_name.
        db: Scoped database session.

    Returns:
        TokenResponse: Access token and public user profile.

    Raises:
        HTTPException: 400 if email is already registered.
    """
    clean_email = payload.email.strip().lower()

    # Check for existing email
    stmt = select(User).where(User.email == clean_email)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists.",
        )

    # Hash password with bcrypt
    hashed_pwd = get_password_hash(payload.password)

    user = User(
        email=clean_email,
        hashed_password=hashed_pwd,
        full_name=payload.full_name.strip() if payload.full_name else None,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info("Registered new user: %s (id=%s)", user.email, user.id)

    # Generate JWT token
    token = create_access_token(subject=user.id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticate with email and password to receive a JWT access token.",
)
async def login(
    payload: UserLogin,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """Authenticate credentials and generate a JWT access token.

    Args:
        payload: Login credentials.
        db: Scoped database session.

    Returns:
        TokenResponse: Access token and public user profile.

    Raises:
        HTTPException: 401 if credentials are invalid, 403 if account is inactive.
    """
    clean_email = payload.email.strip().lower()

    stmt = select(User).where(User.email == clean_email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    logger.info("User logged in successfully: %s", user.email)
    token = create_access_token(subject=user.id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Current User Profile",
    description="Retrieve the profile of the currently authenticated user.",
)
async def get_current_user_profile(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Return profile data for the authenticated caller."""
    return UserResponse.model_validate(current_user)
