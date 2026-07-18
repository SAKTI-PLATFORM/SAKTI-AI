"""CV parsing API for first-stage job seeker onboarding."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.core.exceptions import UnprocessableException
from src.core.http import HTTPDataResponse
from src.domain.cvparser.cvparser import CVParser
from src.domain.cvparser.pdf_extractor import MAX_CV_FILE_SIZE, extract_pdf_text

parse_cv_router = APIRouter(prefix="/ml/cv", tags=["cvparser"])
cv_parser = CVParser()
logger = logging.getLogger("uvicorn.error")


class ParseCVRequest(BaseModel):
    text: str = Field(min_length=1)
    file_name: str | None = None


class ParseCVResult(BaseModel):
    confidence_score: float
    personal_info: dict[str, Any]
    educations: list[dict[str, Any]]
    experiences: list[dict[str, Any]]
    projects: list[dict[str, Any]]
    certifications: list[dict[str, Any]]
    skills: list[dict[str, Any]]


@parse_cv_router.post("/parse", response_model=HTTPDataResponse[ParseCVResult])
async def parse_cv(request: ParseCVRequest) -> HTTPDataResponse[ParseCVResult]:
    logger.info(
        "[CVParser API] Plain-text CV received: text_length=%d file_name=%s",
        len(request.text),
        request.file_name,
    )
    try:
        parsed = await cv_parser.parse_cv(request.text)
    except ValueError as exc:
        raise UnprocessableException(str(exc)) from exc

    return HTTPDataResponse[ParseCVResult](
        message="CV parsed successfully",
        data=ParseCVResult(**parsed),
    )


@parse_cv_router.post("/parse-file", response_model=HTTPDataResponse[ParseCVResult])
async def parse_cv_file(
    cv: UploadFile = File(description="PDF CV dengan ukuran maksimal 10 MB"),
) -> HTTPDataResponse[ParseCVResult]:
    if cv.content_type != "application/pdf":
        raise UnprocessableException("CV wajib berupa file PDF.")

    try:
        content = await cv.read(MAX_CV_FILE_SIZE + 1)
        logger.info(
            "[CVParser API] PDF received: file_name=%s size_bytes=%d",
            cv.filename,
            len(content),
        )
        text = await run_in_threadpool(extract_pdf_text, content)
        logger.info(
            "[CVParser API] PDF text extracted: text_length=%d",
            len(text),
        )
        parsed = await cv_parser.parse_cv(text)
    except ValueError as exc:
        raise UnprocessableException(str(exc)) from exc
    finally:
        await cv.close()

    return HTTPDataResponse[ParseCVResult](
        message="PDF CV extracted and parsed successfully",
        data=ParseCVResult(**parsed),
    )


@parse_cv_router.post("/extract", response_model=HTTPDataResponse[ParseCVResult])
async def extract_cv(request: ParseCVRequest) -> HTTPDataResponse[ParseCVResult]:
    return await parse_cv(request)
