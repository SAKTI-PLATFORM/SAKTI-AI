"""Deterministic scoring functions for career matching.

All functions are pure (no I/O, no LLM calls) and return floats in [0, 1].
"""

from __future__ import annotations

from typing import Literal

from src.core.config import settings
from src.domain.shared.skill_evidence import SkillEvidence
from src.domain.shared.user_profile import OceanScores, RiasecScores


# ── Skill-level heuristics ──────────────────────────────

_LEVEL_MAP = {
    "None": 0,
    "Beginner": 1,
    "Intermediate": 2,
    "Advanced": 3,
}


def determine_skill_level(
    skill: SkillEvidence,
) -> Literal["None", "Beginner", "Intermediate", "Advanced"]:
    """Infer a skill proficiency level from evidence signals.

    Heuristic:
    - Advanced  : confidence ≥ 0.8 AND (working_hours ≥ 1000 OR evidence_strength == "high")
    - Intermediate: confidence ≥ 0.6 AND (working_hours ≥ 200 OR evidence_strength in ["medium", "high"])
    - Beginner  : confidence ≥ 0.3
    - None      : everything else
    """
    c = skill.confidence_score
    wh = skill.working_hours or 0
    es = skill.evidence_strength

    if c >= 0.8 and (wh >= 1000 or es == "high"):
        return "Advanced"
    if c >= 0.6 and (wh >= 200 or es in ("medium", "high")):
        return "Intermediate"
    if c >= 0.3:
        return "Beginner"
    return "None"


def determine_required_level(role_level: str) -> Literal["Beginner", "Intermediate", "Advanced"]:
    """Map a role seniority to its typical minimum skill level.

    - Junior  → Beginner
    - Mid     → Intermediate
    - Senior+ → Advanced
    """
    role_lower = role_level.lower()
    if "senior" in role_lower or "lead" in role_lower or "principal" in role_lower:
        return "Advanced"
    if "mid" in role_lower:
        return "Intermediate"
    return "Beginner"


def gap_level(current: str, required: str) -> Literal["Low", "Medium", "High"]:
    """Compute gap severity between current and required skill levels."""
    diff = _LEVEL_MAP.get(required, 0) - _LEVEL_MAP.get(current, 0)
    if diff <= 0:
        return "Low"
    if diff == 1:
        return "Medium"
    return "High"


def gap_priority(gap_lev: str, is_required: bool) -> Literal["Low", "Medium", "High"]:
    """Determine gap priority based on gap level and whether skill is mandatory."""
    if gap_lev == "High" and is_required:
        return "High"
    if gap_lev in ("Medium", "High"):
        return "Medium"
    return "Low"


# ── Per-dimension scoring ───────────────────────────────


def calc_skill_match_score(
    user_skills: list[SkillEvidence],
    role_required_skills: list[str],
) -> tuple[float, list[str], list[str]]:
    """Score how well user skills match role requirements.

    Returns (score, matched_skills, unmatched_skills).
    """
    if not role_required_skills:
        return 1.0, [], []

    user_skill_names = {s.detected_text.lower() for s in user_skills}
    required_lower = [s.lower() for s in role_required_skills]

    matched = [s for s in role_required_skills if s.lower() in user_skill_names]
    unmatched = [s for s in role_required_skills if s.lower() not in user_skill_names]

    # Weighted by confidence score for matched skills
    if matched:
        skill_map = {s.detected_text.lower(): s for s in user_skills}
        weighted_sum = sum(
            skill_map[m.lower()].confidence_score
            for m in matched
            if m.lower() in skill_map
        )
        score = weighted_sum / len(required_lower)
    else:
        score = 0.0

    return min(score, 1.0), matched, unmatched


def calc_experience_score(
    experiences: list[dict],
    role_min_months: int,
) -> float:
    """Score based on total relevant work experience duration."""
    total_months = sum(
        e.get("duration_months", 0) or 0
        for e in experiences
        if e.get("experience_type") in ("WORK", "INTERNSHIP")
    )
    if role_min_months <= 0:
        return 1.0
    return min(total_months / role_min_months, 1.0)


def calc_education_score(
    educations: list[dict],
    role_level: str,
) -> float:
    """Score education alignment with role requirements."""
    if not educations:
        return 0.3  # Minimum score if no education data

    level_scores = {"SMA": 0.3, "D3": 0.5, "S1": 0.7, "S2": 0.9, "S3": 1.0}
    best = max(
        level_scores.get(e.get("education_level", ""), 0.3)
        for e in educations
    )
    # GPA bonus
    gpa_bonus = 0
    for e in educations:
        gpa = e.get("gpa")
        if gpa and gpa >= 3.5:
            gpa_bonus = 0.1
            break

    return min(best + gpa_bonus, 1.0)


def calc_riasec_fit(
    user_riasec: RiasecScores | None,
    role_ideal_code: str | None,
) -> float:
    """Score RIASEC interest fit between user and role.

    Compares the user's dominant dimensions against the role's ideal code.
    """
    if not user_riasec or not role_ideal_code:
        return 0.5  # Neutral when data unavailable

    dimension_map = {
        "R": user_riasec.realistic,
        "I": user_riasec.investigative,
        "A": user_riasec.artistic,
        "S": user_riasec.social,
        "E": user_riasec.enterprising,
        "C": user_riasec.conventional,
    }

    ideal_dims = list(role_ideal_code.upper()[:3])
    if not ideal_dims:
        return 0.5

    total = sum(dimension_map.get(d, 0) for d in ideal_dims)
    max_possible = 100 * len(ideal_dims)

    return total / max_possible if max_possible > 0 else 0.5


def calc_ocean_workstyle(
    user_ocean: OceanScores | None,
) -> float:
    """Score OCEAN work-style alignment.

    Higher conscientiousness and openness are generally positive.
    """
    if not user_ocean:
        return 0.5

    # Weighted average favoring conscientiousness and openness
    score = (
        user_ocean.openness * 0.3
        + user_ocean.conscientiousness * 0.3
        + user_ocean.extraversion * 0.15
        + user_ocean.agreeableness * 0.15
        + (100 - user_ocean.neuroticism) * 0.1  # Lower neuroticism is better
    )
    return score / 100


def calc_preference_score(
    has_matching_field: bool,
    has_matching_role: bool,
) -> float:
    """Score based on user's explicit career preferences alignment."""
    score = 0.5  # Base
    if has_matching_field:
        score += 0.25
    if has_matching_role:
        score += 0.25
    return score


def calc_total_score(
    scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted total match score (0-100 scale).

    Parameters
    ----------
    scores : dict
        Keys: "skill", "experience", "education", "riasec", "ocean", "preference"
    weights : dict, optional
        Override default weights from config.
    """
    w = weights or settings.jobmatcher_scoring_weights
    total = sum(scores.get(k, 0) * w.get(k, 0) for k in w)
    return round(total * 100, 2)
