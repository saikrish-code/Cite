"""Pydantic validation schemas for user registration, authentication, and tokens."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    """Payload schema for user registration / signup."""

    model_config = ConfigDict(frozen=True)

    email: EmailStr = Field(..., description="Unique email address used for login")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Plaintext password (minimum 8 characters)",
    )
    full_name: str | None = Field(
        default=None,
        max_length=128,
        description="Optional full name or display name",
    )


class UserLogin(BaseModel):
    """Payload schema for user authentication / login."""

    model_config = ConfigDict(frozen=True)

    email: EmailStr = Field(..., description="Registered account email")
    password: str = Field(..., min_length=1, description="Account password")


class UserResponse(BaseModel):
    """Public user profile data returned after registration, login, or /me probe."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique user ID (UUID)")
    email: str = Field(..., description="User's registered email address")
    full_name: str | None = Field(
        default=None, description="User's full name or display alias"
    )
    is_active: bool = Field(
        default=True, description="Whether the account is currently active"
    )
    created_at: datetime = Field(..., description="Account creation timestamp")


class TokenResponse(BaseModel):
    """JWT bearer token response returned upon successful authentication."""

    model_config = ConfigDict(frozen=True)

    access_token: str = Field(..., description="Signed JWT bearer access token")
    token_type: str = Field(
        default="bearer", description="Token authentication type"
    )
    user: UserResponse = Field(
        ..., description="Associated authenticated user profile"
    )
