"""Internal Double Diamond API used only by the NestJS backend."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, status

from src.core.security import require_internal_api_key
from src.domain.double_diamond.schema import (
    AnalyzePhaseRequest,
    AnalyzePhaseResponse,
    GenerateQuestionRequest,
    GenerateQuestionResponse,
)
from src.domain.double_diamond.service import DoubleDiamondService

router = APIRouter(
    prefix="/internal/v1/double-diamond",
    tags=["double-diamond-internal"],
)
service = DoubleDiamondService()
logger = logging.getLogger("uvicorn.error")


@router.post("/questions/generate", response_model=GenerateQuestionResponse)
async def generate_questions(
    request: GenerateQuestionRequest,
    _: None = Depends(require_internal_api_key),
    x_request_id: str | None = Header(default=None),
) -> GenerateQuestionResponse:
    try:
        result = await service.generate_questions(request)
        logger.info(
            "DD questions generated request_id=%s session=%s phase=%s model=%s",
            x_request_id,
            request.onboarding_session_id,
            request.phase,
            result.model_name,
        )
        return result
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM timeout",
        ) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider unavailable",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI structured output invalid",
        ) from exc


@router.post("/phases/analyze", response_model=AnalyzePhaseResponse)
async def analyze_phase(
    request: AnalyzePhaseRequest,
    _: None = Depends(require_internal_api_key),
    x_request_id: str | None = Header(default=None),
) -> AnalyzePhaseResponse:
    try:
        result = await service.analyze_phase(request)
        logger.info(
            "DD phase analyzed request_id=%s session=%s phase=%s model=%s",
            x_request_id,
            request.onboarding_session_id,
            request.phase,
            service.provider.model_name,
        )
        return result
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM timeout",
        ) from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider unavailable",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI structured output invalid",
        ) from exc
