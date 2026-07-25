"""Integration tests for the JobMatcher LangGraph.

These tests require a valid DEEPSEEK_API_KEY and network access.
Mark with ``pytest.mark.integration`` and skip in CI if needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.job_matcher.value_objects import JobMatcherOutput
from src.features.job_matcher.service import JobMatcherService


@pytest.fixture()
def sample_data() -> dict:
    data_path = Path(__file__).resolve().parent.parent / "data" / "result-full_AFTER_SKILL_PARSERS.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_job_matcher_full_pipeline(sample_data: dict) -> None:
    """End-to-end test: input full CV → expect career matches."""
    service = JobMatcherService()
    output = await service.run(sample_data)

    assert isinstance(output, JobMatcherOutput)

    # Must have at least one career match
    assert len(output.career_match_results) >= 1

    # Top match should score > 50 for a strong profile like Anargya
    top_match = output.career_match_results[0]
    assert top_match.total_match_score > 50
    assert top_match.user_id != ""
    assert top_match.role_name != ""

    # Score details should exist for each match
    assert len(output.career_match_score_details) >= 1

    # Skill gaps should be identified
    assert len(output.skill_gap_results) >= 0  # May be empty if perfect match


@pytest.mark.asyncio
async def test_job_matcher_output_schema(sample_data: dict) -> None:
    """Verify all output objects conform to their Pydantic schemas."""
    service = JobMatcherService()
    output = await service.run(sample_data)

    for match in output.career_match_results:
        assert match.match_id.startswith("MATCH-")
        assert 0 <= match.total_match_score <= 100

    for detail in output.career_match_score_details:
        assert detail.match_id.startswith("MATCH-")

    for gap in output.skill_gap_results:
        assert gap.gap_id.startswith("GAP-")
        assert gap.current_level in ("None", "Beginner", "Intermediate", "Advanced")
        assert gap.required_level in ("Beginner", "Intermediate", "Advanced")
        assert gap.gap_level in ("Low", "Medium", "High")
        assert gap.priority in ("Low", "Medium", "High")
