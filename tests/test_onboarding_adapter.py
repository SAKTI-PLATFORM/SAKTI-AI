"""Unit tests for the onboarding data adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.adapters.onboarding_adapter import parse_onboarding_data


@pytest.fixture()
def sample_data() -> dict:
    """Load the sample onboarding export."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "result-full_AFTER_SKILL_PARSERS.json"
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def test_parse_onboarding_data_returns_user_profile(sample_data: dict) -> None:
    profile = parse_onboarding_data(sample_data)

    assert profile.user_id != ""
    assert profile.onboarding_session_id != ""
    assert len(profile.skills) > 0


def test_parse_skills_have_confidence(sample_data: dict) -> None:
    profile = parse_onboarding_data(sample_data)

    for skill in profile.skills:
        assert 0 <= skill.confidence_score <= 1
        assert skill.detected_text != ""
        assert skill.evidence_strength in ("low", "medium", "high")


def test_parse_ocean_scores(sample_data: dict) -> None:
    profile = parse_onboarding_data(sample_data)

    assert profile.ocean is not None
    assert 0 <= profile.ocean.openness <= 100
    assert 0 <= profile.ocean.conscientiousness <= 100


def test_parse_riasec_scores(sample_data: dict) -> None:
    profile = parse_onboarding_data(sample_data)

    assert profile.riasec is not None
    assert profile.riasec.dominant_code != ""


def test_parse_detected_fields(sample_data: dict) -> None:
    profile = parse_onboarding_data(sample_data)

    assert len(profile.detected_fields) > 0
    assert profile.detected_fields[0].code != ""
    assert 0 <= profile.detected_fields[0].score <= 1


def test_parse_recommended_roles(sample_data: dict) -> None:
    profile = parse_onboarding_data(sample_data)

    assert len(profile.recommended_roles) > 0


def test_parse_preferences(sample_data: dict) -> None:
    profile = parse_onboarding_data(sample_data)

    assert profile.preferences is not None
    assert profile.preferences.selected_field is not None


def test_parse_strengths_barriers(sample_data: dict) -> None:
    profile = parse_onboarding_data(sample_data)

    assert len(profile.strengths) > 0
    assert len(profile.barriers) > 0
