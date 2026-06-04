"""Typed HTTP exceptions raised by feature/domain code."""

from __future__ import annotations

from fastapi import HTTPException, status


class BaseHTTPException(HTTPException):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(status_code=status_code, detail=message)


class BadRequestException(BaseHTTPException):
    def __init__(self, message: str = "Bad request") -> None:
        super().__init__(status.HTTP_400_BAD_REQUEST, message)


class NotFoundException(BaseHTTPException):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(status.HTTP_404_NOT_FOUND, message)


class UnprocessableException(BaseHTTPException):
    def __init__(self, message: str = "Unprocessable entity") -> None:
        super().__init__(status.HTTP_422_UNPROCESSABLE_ENTITY, message)


class ServerException(BaseHTTPException):
    def __init__(self, message: str = "Internal server error") -> None:
        super().__init__(status.HTTP_500_INTERNAL_SERVER_ERROR, message)
