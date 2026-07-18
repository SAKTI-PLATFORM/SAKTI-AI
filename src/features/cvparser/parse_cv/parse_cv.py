"""CV parsing API for first-stage job seeker onboarding."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.exceptions import UnprocessableException
from src.core.http import HTTPDataResponse
from src.domain.cvparser.cvparser import CVParser

parse_cv_router = APIRouter(prefix="/ml/cv", tags=["cvparser"])
cv_parser = CVParser()


class ParseCVRequest(BaseModel):
    text: str = Field(min_length=1)
    file_name: str | None = None


class ParseCVResult(BaseModel):
    confidence_score: float
    educations: list[dict[str, Any]]
    experiences: list[dict[str, Any]]
    projects: list[dict[str, Any]]
    certifications: list[dict[str, Any]]
    skills: list[dict[str, Any]]


@parse_cv_router.post("/parse", response_model=HTTPDataResponse[ParseCVResult])
async def parse_cv(request: ParseCVRequest) -> HTTPDataResponse[ParseCVResult]:
    try:
        parsed = await cv_parser.parse_cv(request.text)
    except ValueError as exc:
        raise UnprocessableException(str(exc)) from exc

    return HTTPDataResponse[ParseCVResult](
        message="CV parsed successfully",
        data=ParseCVResult(**parsed),
    )


@parse_cv_router.post("/extract", response_model=HTTPDataResponse[ParseCVResult])
async def extract_cv(request: ParseCVRequest) -> HTTPDataResponse[ParseCVResult]:
    return await parse_cv(request)
