"""Unit tests for deterministic scoring functions."""

from __future__ import annotations

import pytest

from src.domain.shared.skill_evidence import SkillEvidence
from src.domain.shared.user_profile import OceanScores, RiasecScores
from src.infrastructure.tools.scoring import (
    calc_education_score,
    calc_experience_score,
    calc_ocean_workstyle,
    calc_preference_score,
    calc_riasec_fit,
    calc_skill_match_score,
    calc_total_score,
    determine_skill_level,
    gap_level,
    gap_priority,
)


# ── determine_skill_level ──────────────────────────────


def test_skill_level_advanced() -> None:
    skill = SkillEvidence(
        detected_text="Python",
        confidence_score=0.9,
        working_hours=1500,
        evidence_strength="high",
    )
    assert determine_skill_level(skill) == "Advanced"


def test_skill_level_intermediate() -> None:
    skill = SkillEvidence(
        detected_text="SQL",
        confidence_score=0.7,
        working_hours=500,
        evidence_strength="medium",
    )
    assert determine_skill_level(skill) == "Intermediate"


def test_skill_level_beginner() -> None:
    skill = SkillEvidence(
        detected_text="Tableau",
        confidence_score=0.4,
        working_hours=10,
        evidence_strength="low",
    )
    assert determine_skill_level(skill) == "Beginner"


def test_skill_level_none() -> None:
    skill = SkillEvidence(
        detected_text="Rust",
        confidence_score=0.1,
        evidence_strength="low",
    )
    assert determine_skill_level(skill) == "None"


# ── gap_level ──────────────────────────────────────────


def test_gap_level_no_gap() -> None:
    assert gap_level("Advanced", "Advanced") == "Low"
    assert gap_level("Intermediate", "Beginner") == "Low"


def test_gap_level_medium() -> None:
    assert gap_level("Beginner", "Intermediate") == "Medium"


def test_gap_level_high() -> None:
    assert gap_level("None", "Intermediate") == "High"
    assert gap_level("Beginner", "Advanced") == "High"


# ── gap_priority ───────────────────────────────────────


def test_gap_priority_high_required() -> None:
    assert gap_priority("High", is_required=True) == "High"


def test_gap_priority_medium() -> None:
    assert gap_priority("Medium", is_required=True) == "Medium"
    assert gap_priority("High", is_required=False) == "Medium"


def test_gap_priority_low() -> None:
    assert gap_priority("Low", is_required=True) == "Low"


# ── calc_skill_match_score ─────────────────────────────


def test_skill_match_full_match() -> None:
    skills = [
        SkillEvidence(detected_text="Python", confidence_score=0.9, evidence_strength="high"),
        SkillEvidence(detected_text="SQL", confidence_score=0.8, evidence_strength="medium"),
    ]
    score, matched, unmatched = calc_skill_match_score(skills, ["Python", "SQL"])
    assert score > 0.75
    assert len(matched) == 2
    assert len(unmatched) == 0


def test_skill_match_partial() -> None:
    skills = [
        SkillEvidence(detected_text="Python", confidence_score=0.9, evidence_strength="high"),
    ]
    score, matched, unmatched = calc_skill_match_score(skills, ["Python", "SQL", "Tableau"])
    assert 0 < score < 1
    assert "Python" in matched
    assert "SQL" in unmatched


def test_skill_match_empty_requirements() -> None:
    skills = [SkillEvidence(detected_text="Python", confidence_score=0.9, evidence_strength="high")]
    score, _, _ = calc_skill_match_score(skills, [])
    assert score == 1.0


# ── calc_experience_score ──────────────────────────────


def test_experience_score_exceeds() -> None:
    exps = [{"duration_months": 24, "experience_type": "WORK"}]
    assert calc_experience_score(exps, 12) == 1.0


def test_experience_score_partial() -> None:
    exps = [{"duration_months": 6, "experience_type": "WORK"}]
    score = calc_experience_score(exps, 12)
    assert score == pytest.approx(0.5)


# ── calc_education_score ───────────────────────────────


def test_education_score_s1_with_high_gpa() -> None:
    edus = [{"education_level": "S1", "gpa": 3.8}]
    score = calc_education_score(edus, "Junior")
    assert score == pytest.approx(0.8)


# ── calc_riasec_fit ────────────────────────────────────


def test_riasec_fit_perfect() -> None:
    riasec = RiasecScores(
        realistic=90, investigative=85, artistic=80,
        social=60, enterprising=50, conventional=40,
        dominant_code="RIA",
    )
    score = calc_riasec_fit(riasec, "RIA")
    assert score > 0.8


def test_riasec_fit_none() -> None:
    score = calc_riasec_fit(None, "RIA")
    assert score == 0.5


# ── calc_ocean_workstyle ───────────────────────────────


def test_ocean_score_high() -> None:
    ocean = OceanScores(
        openness=80, conscientiousness=85,
        extraversion=70, agreeableness=75,
        neuroticism=20,
    )
    score = calc_ocean_workstyle(ocean)
    assert score > 0.7


def test_ocean_score_none() -> None:
    assert calc_ocean_workstyle(None) == 0.5


# ── calc_total_score ───────────────────────────────────


def test_total_score_all_perfect() -> None:
    scores = {
        "skill": 1.0, "experience": 1.0, "education": 1.0,
        "riasec": 1.0, "ocean": 1.0, "preference": 1.0,
    }
    total = calc_total_score(scores)
    assert total == 100.0


def test_total_score_weighted() -> None:
    scores = {
        "skill": 0.5, "experience": 0.5, "education": 0.5,
        "riasec": 0.5, "ocean": 0.5, "preference": 0.5,
    }
    total = calc_total_score(scores)
    assert total == pytest.approx(50.0)
