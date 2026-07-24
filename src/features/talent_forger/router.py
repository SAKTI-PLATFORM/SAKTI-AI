"""FastAPI router for the TalentForger endpoint."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.features.talent_forger.service import TalentForgerService

logger = logging.getLogger("uvicorn.error")

talent_forger_router = APIRouter(prefix="/api/v1", tags=["talent-forge"])

_service = TalentForgerService()


class TalentForgeRequest(BaseModel):
    """Request body for the TalentForger endpoint."""

    match_id: str
    skill_gaps: list[dict[str, Any]]
    onboarding_data: dict[str, Any]


@talent_forger_router.post("/talent-forge")
async def talent_forge(request: TalentForgeRequest) -> dict[str, Any]:
    """Run the TalentForger pipeline to generate learning paths.

    Expects:
    - match_id: from JobMatcher output
    - skill_gaps: from JobMatcher output
    - onboarding_data: the full onboarding export JSON
    """
    try:
        output = await _service.run(
            match_id=request.match_id,
            skill_gaps=request.skill_gaps,
            raw_input=request.onboarding_data,
        )
        return {
            "status": "success",
            "data": output.model_dump(),
        }
    except Exception:
        logger.exception("[TalentForger Router] Pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Learning path generation failed. Check server logs.",
        )
