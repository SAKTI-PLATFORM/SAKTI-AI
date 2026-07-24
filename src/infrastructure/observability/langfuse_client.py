"""Langfuse callback handler factory for LangGraph graph tracing."""

from __future__ import annotations

import logging

from src.core.config import settings

logger = logging.getLogger("uvicorn.error")


def get_langfuse_handler(
    session_id: str,
    user_id: str,
    module: str,
    *,
    trace_name: str | None = None,
):
    """Create a Langfuse callback handler for a graph run.

    Parameters
    ----------
    session_id : str
        The onboarding session ID (groups related traces).
    user_id : str
        The user being processed.
    module : str
        Module name: ``"JobMatcher"`` or ``"TalentForger"``.
    trace_name : str, optional
        Custom trace name, defaults to ``"{module}_run"``.

    Returns
    -------
    CallbackHandler or None
        Returns ``None`` if Langfuse is not configured (dev mode).
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "[Langfuse] Not configured — tracing disabled for module=%s",
            module,
        )
        return None

    try:
        from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

        handler = LangfuseCallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name or f"{module}_run",
            tags=[f"module:{module}"],
        )
        logger.info(
            "[Langfuse] Handler created: module=%s session=%s",
            module,
            session_id,
        )
        return handler
    except Exception:
        logger.exception("[Langfuse] Failed to create handler")
        return None
