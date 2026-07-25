"""Authentication for service-to-service endpoints."""

from __future__ import annotations

from hmac import compare_digest

from fastapi import Header, HTTPException, status

from src.core.config import settings


async def require_internal_api_key(
    x_internal_api_key: str | None = Header(default=None),
) -> None:
    if not x_internal_api_key or not compare_digest(
        x_internal_api_key,
        settings.internal_api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key",
        )
