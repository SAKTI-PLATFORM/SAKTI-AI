"""Transform raw onboarding JSON export into UserProfileInput."""

from __future__ import annotations

from src.domain.shared.skill_evidence import SkillEvidence
from src.domain.shared.user_profile import (
    DetectedField,
    OceanScores,
    RecommendedRole,
    RiasecScores,
    UserPreferences,
    UserProfileInput,
)


def parse_onboarding_data(raw: dict) -> UserProfileInput:
    """Convert the full onboarding export JSON into a unified profile.

    Expects the structure from ``result-full_AFTER_SKILL_PARSERS.json``:
    ``{"data": {"session": {…}, "cv": {…}, "profile": {…}, "assessment": {…}, "double_diamond": {…}}}``
    """
    data = raw.get("data", raw)

    session = data.get("session", {})
    profile = data.get("profile", {})
    identity = profile.get("identity", {})
    assessment = data.get("assessment", {})
    dd = data.get("double_diamond", {})
    dd_result = dd.get("result", {})
    cv_data = data.get("cv", {})

    # ── Skills (prefer profile.skills with IDs, fallback to cv parsed) ─
    profile_skills = profile.get("skills", [])
    if not profile_skills:
        parsed = cv_data.get("parsed_result", {})
        profile_skills = parsed.get("skills", [])

    skills = [
        SkillEvidence(
            detected_text=s.get("detected_text", ""),
            confidence_score=s.get("confidence_score", 0.5),
            learning_hours=s.get("learning_hours"),
            working_hours=s.get("working_hours"),
            evidence_source=s.get("evidence_source", "cv_text"),
            evidence_strength=s.get("evidence_strength", "medium"),
            user_skill_id=s.get("user_skill_id"),
            skill_id=s.get("skill_id"),
        )
        for s in profile_skills
        if s.get("detected_text")
    ]

    # ── OCEAN & RIASEC ──────────────────────────────────
    assess_result = assessment.get("result", {})
    ocean_raw = assess_result.get("ocean")
    riasec_raw = assess_result.get("riasec")

    ocean = OceanScores(**ocean_raw) if ocean_raw else None
    riasec = RiasecScores(**riasec_raw) if riasec_raw else None

    # ── Double Diamond ──────────────────────────────────
    detected_fields = [
        DetectedField(**f) for f in dd_result.get("detected_fields", [])
    ]
    recommended_roles = [
        RecommendedRole(**r) for r in dd_result.get("recommended_roles", [])
    ]

    # ── Preferences from DD questions ───────────────────
    dd_questions = dd.get("questions", [])
    preferences = _extract_preferences(dd_questions, dd_result)

    return UserProfileInput(
        user_id=identity.get("user_id", ""),
        cv_id=cv_data.get("cv_id", session.get("cv_id", "")),
        onboarding_session_id=session.get("onboarding_session_id", ""),
        full_name=identity.get("full_name", ""),
        professional_headline=identity.get("professional_headline"),
        profile_summary=identity.get("profile_summary"),
        skills=skills,
        educations=profile.get("educations", []),
        experiences=profile.get("experiences", []),
        projects=profile.get("projects", []),
        certifications=profile.get("certifications", []),
        ocean=ocean,
        riasec=riasec,
        preferences=preferences,
        detected_fields=detected_fields,
        recommended_roles=recommended_roles,
        strengths=dd_result.get("strengths", []),
        barriers=dd_result.get("barriers", []),
        career_summary=dd_result.get("career_summary"),
        work_style_summary=dd_result.get("work_style_summary"),
    )


def _extract_preferences(
    questions: list[dict], dd_result: dict
) -> UserPreferences:
    """Extract career preferences from Double Diamond answers."""
    answers: dict[str, str] = {}
    for q in questions:
        code = q.get("question_code", "")
        answer = q.get("answer")
        if code and answer:
            answers[code] = str(answer) if not isinstance(answer, list) else ",".join(answer)

    return UserPreferences(
        career_goal=dd_result.get("career_summary"),
        selected_field=dd_result.get("selected_field"),
        selected_role=dd_result.get("selected_role"),
        learning_style=answers.get("Q4_LEARNING_STYLE"),
    )
