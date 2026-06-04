"""SAKTI Lens — POST /ml/score-psychometric.

Stateless: receives the Step-03 onboarding payload (OCEAN + RIASEC raw responses)
and returns the computed trait vectors, confidence, and Holland code. NestJS caches
the result into seeker_profiles for fast dashboard render.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.core.exceptions import UnprocessableException
from src.core.http import HTTPDataResponse
from src.domain.psychometric.ocean import (
    OceanResponse,
    OceanTrait,
    Polarity,
    score_ocean,
)
from src.domain.psychometric.riasec import RiasecResponse, score_riasec

score_psychometric_router = APIRouter(prefix="/ml", tags=["psychometric"])


class OceanResponseDto(BaseModel):
    trait: OceanTrait
    polarity: Polarity = Field(description="'+' positive item, '-' reverse-scored item")
    value: int = Field(ge=1, le=7, description="Likert 1-7")


class RiasecResponseDto(BaseModel):
    item: int = Field(ge=1, le=42)
    agreed: bool


class ScorePsychometricRequest(BaseModel):
    ocean_responses: list[OceanResponseDto]
    riasec_responses: list[RiasecResponseDto]


class OceanResultDto(BaseModel):
    scores: dict[str, float]
    trait_confidence: dict[str, float]
    confidence: float


class RiasecResultDto(BaseModel):
    scores: dict[str, float]
    raw: dict[str, int]
    holland_code: str


class PsychometricResultDto(BaseModel):
    ocean: OceanResultDto
    riasec: RiasecResultDto


@score_psychometric_router.post(
    "/score-psychometric",
    response_model=HTTPDataResponse[PsychometricResultDto],
)
async def score_psychometric(
    request: ScorePsychometricRequest,
) -> HTTPDataResponse[PsychometricResultDto]:
    try:
        ocean = score_ocean(
            [
                OceanResponse(trait=item.trait, polarity=item.polarity, value=item.value)
                for item in request.ocean_responses
            ]
        )
        riasec = score_riasec(
            [
                RiasecResponse(item=item.item, agreed=item.agreed)
                for item in request.riasec_responses
            ]
        )
    except ValueError as exc:
        raise UnprocessableException(str(exc)) from exc

    return HTTPDataResponse[PsychometricResultDto](
        message="Psychometric scored successfully",
        data=PsychometricResultDto(
            ocean=OceanResultDto(
                scores=ocean.scores,
                trait_confidence=ocean.trait_confidence,
                confidence=ocean.confidence,
            ),
            riasec=RiasecResultDto(
                scores=riasec.scores,
                raw=riasec.raw_scores,
                holland_code=riasec.holland_code,
            ),
        ),
    )
