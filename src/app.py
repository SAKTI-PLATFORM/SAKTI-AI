"""SAKTI-AI — FastAPI ML inference service (OCR, embed, psychometric, RAG, match)."""

from fastapi import FastAPI

from src.core.error_handler import add_global_exception_handlers
from src.features.insight.generate_insight.generate_insight import (
    generate_insight_router,
)
from src.features.psychometric.score_psychometric.score_psychometric import (
    score_psychometric_router,
)

app = FastAPI(title="SAKTI-AI", description="SAKTI ML inference service")
add_global_exception_handlers(app)

app.include_router(score_psychometric_router)
app.include_router(generate_insight_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
