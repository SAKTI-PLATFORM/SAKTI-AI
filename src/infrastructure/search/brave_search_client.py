"""Brave LLM Context API client for grounded web search.

Wraps the Brave Search LLM Context endpoint:
    POST https://api.search.brave.com/res/v1/llm/context

Returns pre-extracted web content ready for LLM consumption.
No scraping needed — Brave delivers ranked text chunks directly.

Reference: https://api.search.brave.com/app/documentation/llm-context
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import httpx

from src.core.config import settings

logger = logging.getLogger("uvicorn.error")

BRAVE_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"

# ── Response models ────────────────────────────────────


@dataclass
class BraveGroundingItem:
    """A single grounded URL with extracted snippets."""

    url: str
    title: str
    snippets: list[str] = field(default_factory=list)

    def as_context_text(self) -> str:
        """Flatten snippets into a single readable block for LLM injection."""
        body = "\n".join(f"  - {s}" for s in self.snippets)
        return f"[{self.title}]({self.url}):\n{body}"


@dataclass
class BraveContextResult:
    """Parsed result from the Brave LLM Context API."""

    items: list[BraveGroundingItem] = field(default_factory=list)
    sources: dict[str, dict] = field(default_factory=dict)
    query: str = ""

    def as_combined_context(self, max_items: int = 5) -> str:
        """Combine top grounding items into a single LLM-ready context string."""
        chunks = [item.as_context_text() for item in self.items[:max_items]]
        return "\n\n---\n\n".join(chunks)

    @property
    def source_urls(self) -> list[str]:
        return [item.url for item in self.items]


# ── Client ─────────────────────────────────────────────


class BraveSearchClient:
    """Async HTTP client for the Brave LLM Context API.

    Usage::

        client = BraveSearchClient()
        result = await client.search("Python data engineer jobs Indonesia 2025")
        context_text = result.as_combined_context()
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or settings.brave_api_key
        self._timeout = timeout

        if not self._api_key:
            logger.warning(
                "[BraveSearch] BRAVE_API_KEY not set — web search will be disabled"
            )

    async def search(
        self,
        query: str,
        *,
        country: str = "id",
        search_lang: str = "id",
        count: int = 10,
        freshness: str = "",
        **kwargs
    ) -> BraveContextResult:
        """Execute a web search via Brave Search API.

        Parameters
        ----------
        query : str
            The search query.
        country : str
            2-letter country code for localised results. Defaults to ``"id"``.
        search_lang : str
            Language preference for results. Defaults to ``"id"``.
        count : int
            Max results to consider. Default 10.
        freshness : str
            Freshness filter (``"pd"``, ``"pw"``, ``"pm"``, ``"py"``, or date range).
        """
        if not self._api_key:
            logger.warning("[BraveSearch] Skipping search — no API key: %s", query)
            return BraveContextResult(query=query)

        params: dict = {
            "q": query,
            "country": country,
            "count": count,
        }
        if freshness:
            params["freshness"] = freshness

        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }

        logger.debug("[BraveSearch] Querying: %r (country=%s)", query, country)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers=headers,
            )
            if response.status_code == 422:
                logger.error("[BraveSearch] 422 Error: %s", response.text)
            response.raise_for_status()
            data = response.json()

        return _parse_response(data, query=query)


def _parse_response(data: dict, *, query: str) -> BraveContextResult:
    """Parse Brave Web Search API response into typed dataclasses."""
    web = data.get("web", {})
    results = web.get("results", [])

    items = []
    for item in results:
        url = item.get("url", "")
        title = item.get("title", "")
        description = item.get("description", "")
        extra = item.get("extra_snippets", [])
        
        snippets = []
        if description:
            snippets.append(description)
        if isinstance(extra, list):
            snippets.extend([s for s in extra if isinstance(s, str)])
            
        if url:
            items.append(BraveGroundingItem(url=url, title=title, snippets=snippets))

    return BraveContextResult(items=items, query=query)


# ── Singleton ──────────────────────────────────────────

_brave_client: BraveSearchClient | None = None


def get_brave_client() -> BraveSearchClient:
    """Return a module-level singleton BraveSearchClient."""
    global _brave_client
    if _brave_client is None:
        _brave_client = BraveSearchClient()
    return _brave_client
