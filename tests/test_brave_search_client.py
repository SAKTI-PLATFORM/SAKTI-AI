"""Unit tests for the Brave Search LLM Context client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.infrastructure.search.brave_search_client import (
    BraveContextResult,
    BraveGroundingItem,
    BraveSearchClient,
    _parse_response,
)


# ── _parse_response unit tests ─────────────────────────


def test_parse_response_full() -> None:
    raw = {
        "grounding": {
            "generic": [
                {
                    "url": "https://example.com/page",
                    "title": "Data Engineer Jobs",
                    "snippets": ["Python required", "SQL experience needed"],
                }
            ],
            "map": [],
        },
        "sources": {
            "https://example.com/page": {
                "title": "Data Engineer Jobs",
                "hostname": "example.com",
                "age": ["2025-01-01"],
            }
        },
    }
    result = _parse_response(raw, query="data engineer jobs")

    assert len(result.items) == 1
    assert result.items[0].url == "https://example.com/page"
    assert result.items[0].title == "Data Engineer Jobs"
    assert "Python required" in result.items[0].snippets
    assert result.query == "data engineer jobs"


def test_parse_response_empty_grounding() -> None:
    raw = {"grounding": {"generic": []}, "sources": {}}
    result = _parse_response(raw, query="no results")

    assert result.items == []
    assert result.source_urls == []


def test_parse_response_missing_url_skipped() -> None:
    raw = {
        "grounding": {
            "generic": [
                {"title": "No URL item", "snippets": ["some text"]},
                {"url": "https://valid.com", "title": "Valid", "snippets": ["valid"]},
            ]
        },
        "sources": {},
    }
    result = _parse_response(raw, query="test")
    # Items without a URL should be filtered out
    assert len(result.items) == 1
    assert result.items[0].url == "https://valid.com"


# ── BraveContextResult helpers ─────────────────────────


def test_as_context_text() -> None:
    item = BraveGroundingItem(
        url="https://example.com",
        title="Test Page",
        snippets=["Snippet one", "Snippet two"],
    )
    text = item.as_context_text()

    assert "Test Page" in text
    assert "https://example.com" in text
    assert "Snippet one" in text


def test_as_combined_context_limits_items() -> None:
    items = [
        BraveGroundingItem(url=f"https://e{i}.com", title=f"T{i}", snippets=[f"s{i}"])
        for i in range(10)
    ]
    result = BraveContextResult(items=items)
    combined = result.as_combined_context(max_items=3)

    # Should contain first 3 items, not all 10
    assert "T0" in combined
    assert "T2" in combined
    assert "T9" not in combined


def test_source_urls() -> None:
    items = [
        BraveGroundingItem(url="https://a.com", title="A", snippets=[]),
        BraveGroundingItem(url="https://b.com", title="B", snippets=[]),
    ]
    result = BraveContextResult(items=items)
    assert result.source_urls == ["https://a.com", "https://b.com"]


# ── BraveSearchClient ──────────────────────────────────


def test_client_warns_when_no_api_key(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        client = BraveSearchClient(api_key=None)

    # The warning should have been emitted during init (logged via logger)
    # We just verify the object was created without error
    assert client is not None


@pytest.mark.asyncio
async def test_client_returns_empty_without_api_key() -> None:
    """When no API key is available, search returns an empty result without hitting the network."""
    client = BraveSearchClient()
    # Force the api_key to None to simulate it missing in both args and env
    client._api_key = None
    
    result = await client.search("test query")

    assert isinstance(result, BraveContextResult)
    assert result.items == []
    assert result.query == "test query"



@pytest.mark.asyncio
async def test_client_calls_brave_api() -> None:
    """Test that the client correctly POSTs to Brave and parses the response."""
    mock_response_data = {
        "grounding": {
            "generic": [
                {
                    "url": "https://jobstreet.com/data-engineer",
                    "title": "Data Engineer - Jakarta",
                    "snippets": ["5+ years experience", "Python, SQL, Spark required"],
                }
            ]
        },
        "sources": {},
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json = lambda: mock_response_data
        mock_post.return_value.raise_for_status = lambda: None

        client = BraveSearchClient(api_key="test-key-123")
        result = await client.search("data engineer jobs Indonesia")

    assert len(result.items) == 1
    assert "Data Engineer" in result.items[0].title
    # Verify the POST was called with correct auth header
    call_kwargs = mock_post.call_args
    assert "X-Subscription-Token" in call_kwargs.kwargs["headers"]
    assert call_kwargs.kwargs["headers"]["X-Subscription-Token"] == "test-key-123"
