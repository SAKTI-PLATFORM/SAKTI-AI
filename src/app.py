"""SAKTI-AI - FastAPI service for CV parsing and extraction."""

from fastapi import FastAPI

from src.core.error_handler import add_global_exception_handlers
from src.features.cvparser.parse_cv.parse_cv import parse_cv_router

app = FastAPI(title="SAKTI-AI", description="SAKTI CV parsing service")
add_global_exception_handlers(app)

app.include_router(parse_cv_router)


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
