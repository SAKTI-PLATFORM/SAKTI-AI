"""Standard HTTP envelope models shared by every feature router."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class HTTPDataResponse(BaseModel, Generic[T]):
    status: str = "success"
    message: str
    data: T


class HTTPMessageResponse(BaseModel):
    status: str = "success"
    message: str


class HTTPErrorResponse(BaseModel):
    status: str = "error"
    error: str
    errors: list[dict[str, Any] | str] = []
