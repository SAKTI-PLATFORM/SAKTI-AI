"""
Optional Gemini text generation. Returns None (graceful fallback to the grounded
template) whenever the API key or the SDK is unavailable, so the service never
hard-depends on an external LLM for the demo.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def generate_text(prompt: str) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        logger.warning("google-generativeai not installed; using grounded template.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_MODEL)
        response = model.generate_content(prompt)
        text = (getattr(response, "text", "") or "").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 - any LLM error → safe fallback
        logger.warning("Gemini generation failed (%s); using grounded template.", exc)
        return None
