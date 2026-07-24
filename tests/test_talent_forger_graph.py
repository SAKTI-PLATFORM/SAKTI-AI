"""Integration tests for the TalentForger LangGraph.

These tests require a valid DEEPSEEK_API_KEY and network access.
Mark with ``pytest.mark.integration`` and skip in CI if needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.talent_forger.value_objects import TalentForgerOutput
from src.features.talent_forger.service import TalentForgerService


@pytest.fixture()
def sample_data() -> dict:
    data_path = Path(__file__).resolve().parent.parent / "data" / "result-full_AFTER_SKILL_PARSERS.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def sample_skill_gaps() -> list[dict]:
    """Sample skill gaps from a JobMatcher run."""
    return [
        {
            "gap_id": "GAP-TEST0001",
            "match_id": "MATCH-TEST001",
            "skill_name": "MLOps",
            "current_level": "None",
            "required_level": "Intermediate",
            "gap_level": "High",
            "priority": "High",
            "reason": "",
        },
        {
            "gap_id": "GAP-TEST0002",
            "match_id": "MATCH-TEST001",
            "skill_name": "Deep Learning",
            "current_level": "Beginner",
            "required_level": "Intermediate",
            "gap_level": "Medium",
            "priority": "Medium",
            "reason": "",
        },
    ]


@pytest.mark.asyncio
async def test_talent_forger_full_pipeline(
    sample_data: dict, sample_skill_gaps: list[dict]
) -> None:
    """End-to-end test: skill gaps → learning path generation."""
    service = TalentForgerService()
    output = await service.run(
        match_id="MATCH-TEST001",
        skill_gaps=sample_skill_gaps,
        raw_input=sample_data,
    )

    assert isinstance(output, TalentForgerOutput)

    # Must have at least one learning path
    assert len(output.learning_paths) >= 1

    # Must have steps for the gaps
    assert len(output.learning_path_steps) >= 1

    # Must have learning resources
    assert len(output.learning_resources) >= 1


@pytest.mark.asyncio
async def test_talent_forger_output_schema(
    sample_data: dict, sample_skill_gaps: list[dict]
) -> None:
    """Verify all output objects conform to their Pydantic schemas."""
    service = TalentForgerService()
    output = await service.run(
        match_id="MATCH-TEST001",
        skill_gaps=sample_skill_gaps,
        raw_input=sample_data,
    )

    for lp in output.learning_paths:
        assert lp.learning_path_id != ""
        assert lp.match_id == "MATCH-TEST001"
        assert lp.estimated_duration_weeks > 0

    for step in output.learning_path_steps:
        assert step.step_id != ""
        assert step.step_order > 0

    for resource in output.learning_resources:
        assert resource.resource_id != ""
        assert resource.resource_type in ("Course", "Certification", "Article", "Video")
