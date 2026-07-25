"""FastAPI router for the JobMatcher endpoint."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from src.features.job_matcher.service import JobMatcherService

logger = logging.getLogger("uvicorn.error")

job_matcher_router = APIRouter(prefix="/api/v1", tags=["career-match"])

_service = JobMatcherService()


@job_matcher_router.post("/career-match")
async def career_match(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the JobMatcher pipeline on the provided onboarding data.

    Expects the full onboarding export JSON as the request body.

    Returns the career match results, score breakdowns, and skill gap analysis.
    """
    try:
        output = await _service.run(payload)
        return {
            "status": "success",
            "data": output.model_dump(),
        }
    except Exception:
        logger.exception("[JobMatcher Router] Pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Career matching pipeline failed. Check server logs.",
        )
