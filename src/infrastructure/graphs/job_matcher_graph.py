"""JobMatcher LangGraph — career matching pipeline.

Graph flow:
    START → extract_preferences → search_roles → calc_similarity
          → score_dimensions → generate_skill_gap → analyze_market
          → explain_matches → END
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Literal, cast

from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.core.config import settings
from src.domain.job_matcher.schemas import (
    CareerMatchResult,
    MatchExplanation,
    RoleReference,
    ScoreDetail,
    SkillGapResult,
)
from src.domain.job_matcher.value_objects import JobMatcherState
from src.domain.shared.skill_evidence import SkillEvidence
from src.domain.shared.user_profile import UserPreferences
from src.infrastructure.llm.deepseek_client import get_deepseek_llm
from src.infrastructure.tools.scoring import (
    calc_education_score,
    calc_experience_score,
    calc_ocean_workstyle,
    calc_preference_score,
    calc_riasec_fit,
    calc_skill_match_score,
    calc_total_score,
    determine_required_level,
    determine_skill_level,
    gap_level,
    gap_priority,
)
from src.infrastructure.tools.search import (
    analyze_market_demand,
    search_job_roles,
)

logger = logging.getLogger("uvicorn.error")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


# ── Node functions ──────────────────────────────────────


async def extract_preferences(state: JobMatcherState) -> dict:
    """Extract user preferences from the profile data."""
    writer = get_stream_writer()
    writer({"step": 1, "total": 7, "title": "Mengekstrak preferensi karier..."})

    profile = state["user_profile"]
    prefs = profile.get("preferences") if isinstance(profile, dict) else getattr(profile, "preferences", None)

    return {
        "preferences": prefs,
        "progress_step": 1,
    }


async def search_roles(state: JobMatcherState) -> dict:
    """Search for candidate career roles using LLM."""
    writer = get_stream_writer()
    writer({"step": 2, "total": 7, "title": "Mencari role karier yang sesuai..."})

    profile = state["user_profile"]
    if isinstance(profile, dict):
        skills_text = ", ".join(s.get("detected_text", "") for s in profile.get("skills", []))
        summary = profile.get("profile_summary", "") or profile.get("professional_headline", "")
        recommended = profile.get("recommended_roles", [])
        fields = profile.get("detected_fields", [])
    else:
        skills_text = ", ".join(s.detected_text for s in profile.skills)
        summary = profile.profile_summary or profile.professional_headline or ""
        recommended = [r.model_dump() for r in profile.recommended_roles]
        fields = [f.model_dump() for f in profile.detected_fields]

    roles = await search_job_roles(
        profile_summary=summary,
        skills_text=skills_text,
        recommended_roles=recommended,
        detected_fields=fields,
        top_k=5,
    )

    return {
        "candidate_roles": [r.model_dump() for r in roles],
        "progress_step": 2,
    }


async def calc_similarity(state: JobMatcherState) -> dict:
    """Calculate skill overlap similarity for each candidate role."""
    writer = get_stream_writer()
    writer({"step": 3, "total": 7, "title": "Menghitung kecocokan skill..."})

    profile = state["user_profile"]
    if isinstance(profile, dict):
        user_skills = [SkillEvidence(**s) for s in profile.get("skills", [])]
    else:
        user_skills = profile.skills

    results = []
    for role_dict in state.get("candidate_roles", []):
        role = RoleReference(**role_dict) if isinstance(role_dict, dict) else role_dict
        score, matched, unmatched = calc_skill_match_score(
            user_skills, role.required_skills
        )
        results.append({
            "role_id": role.role_id,
            "raw_similarity": score,
            "matched_skills": matched,
            "unmatched_skills": unmatched,
        })

    return {
        "similarity_results": results,
        "progress_step": 3,
    }


async def score_dimensions(state: JobMatcherState) -> dict:
    """Score each candidate role across all dimensions."""
    writer = get_stream_writer()
    writer({"step": 4, "total": 7, "title": "Menghitung skor multi-dimensi..."})

    profile = state["user_profile"]
    if isinstance(profile, dict):
        user_skills = [SkillEvidence(**s) for s in profile.get("skills", [])]
        experiences = profile.get("experiences", [])
        educations = profile.get("educations", [])
        ocean = profile.get("ocean")
        riasec = profile.get("riasec")
        prefs = profile.get("preferences", {})
        selected_field = prefs.get("selected_field") if prefs else None
        selected_role = prefs.get("selected_role") if prefs else None
    else:
        user_skills = profile.skills
        experiences = profile.experiences
        educations = profile.educations
        ocean = profile.ocean
        riasec = profile.riasec
        prefs = profile.preferences
        selected_field = prefs.selected_field if prefs else None
        selected_role = prefs.selected_role if prefs else None

    from src.domain.shared.user_profile import OceanScores, RiasecScores

    if isinstance(ocean, dict):
        ocean = OceanScores(**ocean)
    if isinstance(riasec, dict):
        riasec = RiasecScores(**riasec)

    score_details = []
    career_matches = []

    sim_map = {
        s["role_id"]: s
        for s in state.get("similarity_results", [])
    }

    for role_dict in state.get("candidate_roles", []):
        role = RoleReference(**role_dict) if isinstance(role_dict, dict) else role_dict
        match_id = f"MATCH-{uuid.uuid4().hex[:8].upper()}"

        sim = sim_map.get(role.role_id, {})
        skill_score = sim.get("raw_similarity", 0)
        exp_score = calc_experience_score(experiences, role.min_experience_months)
        edu_score = calc_education_score(educations, role.role_level)
        riasec_score = calc_riasec_fit(riasec, role.riasec_ideal)
        ocean_score = calc_ocean_workstyle(ocean)

        # Check preference alignment
        has_field = bool(
            selected_field and selected_field.lower() in role.role_category.lower()
        )
        has_role = bool(
            selected_role and selected_role.lower() in role.role_name.lower()
        )
        pref_score = calc_preference_score(has_field, has_role)

        scores = {
            "skill": skill_score,
            "experience": exp_score,
            "education": edu_score,
            "riasec": riasec_score,
            "ocean": ocean_score,
            "preference": pref_score,
        }
        total = calc_total_score(scores)

        detail = ScoreDetail(
            match_id=match_id,
            skill_match_score=round(skill_score * 100, 2),
            experience_project_score=round(exp_score * 100, 2),
            education_score=round(edu_score * 100, 2),
            riasec_fit_score=round(riasec_score * 100, 2),
            ocean_workstyle_score=round(ocean_score * 100, 2),
            preference_score=round(pref_score * 100, 2),
        )
        score_details.append(detail.model_dump())

        match = CareerMatchResult(
            match_id=match_id,
            user_id=profile["user_id"] if isinstance(profile, dict) else profile.user_id,
            role_id=role.role_id,
            role_name=role.role_name,
            total_match_score=total,
        )
        career_matches.append(match.model_dump())

    return {
        "score_details": score_details,
        "career_matches": career_matches,
        "progress_step": 4,
    }


async def generate_skill_gap(state: JobMatcherState) -> dict:
    """Generate skill gap analysis for top matches."""
    writer = get_stream_writer()
    writer({"step": 5, "total": 7, "title": "Menganalisis kesenjangan skill..."})

    profile = state["user_profile"]
    if isinstance(profile, dict):
        user_skills = [SkillEvidence(**s) for s in profile.get("skills", [])]
    else:
        user_skills = profile.skills

    skill_level_map = {
        s.detected_text.lower(): determine_skill_level(s)
        for s in user_skills
    }

    gaps = []
    for match_dict in state.get("career_matches", []):
        match_id = match_dict["match_id"]
        role_id = match_dict["role_id"]
        total_score = match_dict["total_match_score"]

        # Find the role
        role_dict = next(
            (r for r in state.get("candidate_roles", []) if r.get("role_id") == role_id),
            None,
        )
        if not role_dict:
            continue

        role = RoleReference(**role_dict) if isinstance(role_dict, dict) else role_dict
        req_level = determine_required_level(role.role_level)

        # Only generate full gaps for matches below threshold
        skills_to_check = role.required_skills
        if total_score >= settings.match_threshold * 100:
            # Only check nice-to-have for high matches
            skills_to_check = role.nice_to_have_skills

        for skill_name in skills_to_check:
            current = skill_level_map.get(skill_name.lower(), "None")
            gl = gap_level(current, req_level)
            is_required = skill_name in role.required_skills

            if gl != "Low":  # Only report meaningful gaps
                gap = SkillGapResult(
                    gap_id=f"GAP-{uuid.uuid4().hex[:8].upper()}",
                    match_id=match_id,
                    skill_name=skill_name,
                    current_level=cast(Literal["None", "Beginner", "Intermediate", "Advanced"], current),
                    required_level=cast(Literal["Beginner", "Intermediate", "Advanced"], req_level),
                    gap_level=cast(Literal["Low", "Medium", "High"], gl),
                    priority=cast(Literal["Low", "Medium", "High"], gap_priority(gl, is_required)),
                )
                gaps.append(gap.model_dump())

    return {
        "skill_gaps": gaps,
        "progress_step": 5,
    }


async def analyze_market(state: JobMatcherState) -> dict:
    """Analyze market demand for gap skills."""
    writer = get_stream_writer()
    writer({"step": 6, "total": 7, "title": "Menganalisis permintaan pasar..."})

    gap_skills = list({
        g["skill_name"] for g in state.get("skill_gaps", [])
    })

    if not gap_skills:
        return {"market_demand": {}, "progress_step": 6}

    demands = await analyze_market_demand(gap_skills[:15])
    demand_map = {d.skill_name: d.demand_score for d in demands}

    return {
        "market_demand": demand_map,
        "progress_step": 6,
    }


async def explain_matches(state: JobMatcherState) -> dict:
    """Use LLM to generate human-readable explanations for matches and gaps."""
    writer = get_stream_writer()
    writer({"step": 7, "total": 7, "title": "Menyusun penjelasan kecocokan..."})

    prompt_template = _load_prompt("job_matcher_explain.txt")

    profile = state["user_profile"]
    if isinstance(profile, dict):
        skills_text = ", ".join(
            f"{s.get('detected_text')} (conf={s.get('confidence_score')}, evidence={s.get('evidence_strength')})"
            for s in profile.get("skills", [])[:20]
        )
    else:
        skills_text = ", ".join(
            f"{s.detected_text} (conf={s.confidence_score}, evidence={s.evidence_strength})"
            for s in profile.skills[:20]
        )

    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(MatchExplanation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "User Skills: {skills_text}\n\n"
            "Career Match: {match_json}\n\n"
            "Score Details: {score_json}\n\n"
            "Skill Gaps: {gaps_json}\n\n"
            "Market Demand: {demand_json}\n\n"
            "Generate match_reason and gap_reasons for match_id={match_id}.",
        ),
    ])

    import json

    updated_matches = []
    updated_gaps = []

    for match_dict in state.get("career_matches", []):
        match_id = match_dict["match_id"]

        # Find related score details and gaps
        score_detail = next(
            (s for s in state.get("score_details", []) if s.get("match_id") == match_id),
            {},
        )
        related_gaps = [
            g for g in state.get("skill_gaps", []) if g.get("match_id") == match_id
        ]

        try:
            chain = prompt | structured_llm
            result = await chain.ainvoke({
                "skills_text": skills_text,
                "match_json": json.dumps(match_dict, default=str),
                "score_json": json.dumps(score_detail, default=str),
                "gaps_json": json.dumps(related_gaps, default=str),
                "demand_json": json.dumps(state.get("market_demand", {})),
                "match_id": match_id,
            })
            explanation = cast(MatchExplanation, result)

            # Update match reason
            match_dict["match_reason"] = explanation.match_reason
            updated_matches.append(match_dict)

            # Update gap reasons
            gap_reason_map = {gr.gap_id: gr.reason for gr in explanation.gap_reasons}
            for g in related_gaps:
                if g["gap_id"] in gap_reason_map:
                    g["reason"] = gap_reason_map[g["gap_id"]]
                updated_gaps.append(g)

        except Exception:
            logger.exception(
                "[JobMatcher] Failed to generate explanation for match=%s", match_id
            )
            match_dict["match_reason"] = "Penjelasan tidak tersedia."
            updated_matches.append(match_dict)
            updated_gaps.extend(related_gaps)

    # We need to overwrite, not append — use the full list
    return {
        "progress_step": 7,
    }


# ── Conditional routing ────────────────────────────────


def should_analyze_market(
    state: JobMatcherState,
) -> Literal["analyze_market", "explain_matches"]:
    """Skip market analysis if no skill gaps were found."""
    if state.get("skill_gaps"):
        return "analyze_market"
    return "explain_matches"


# ── Graph builder ──────────────────────────────────────


def build_job_matcher_graph() -> StateGraph:
    """Build and return the compiled JobMatcher LangGraph."""
    graph = StateGraph(JobMatcherState)

    # Add nodes
    graph.add_node("extract_preferences", extract_preferences)
    graph.add_node("search_roles", search_roles)
    graph.add_node("calc_similarity", calc_similarity)
    graph.add_node("score_dimensions", score_dimensions)
    graph.add_node("generate_skill_gap", generate_skill_gap)
    graph.add_node("analyze_market", analyze_market)
    graph.add_node("explain_matches", explain_matches)

    # Wire edges
    graph.add_edge(START, "extract_preferences")
    graph.add_edge("extract_preferences", "search_roles")
    graph.add_edge("search_roles", "calc_similarity")
    graph.add_edge("calc_similarity", "score_dimensions")
    graph.add_edge("score_dimensions", "generate_skill_gap")
    graph.add_conditional_edges(
        "generate_skill_gap",
        should_analyze_market,
        ["analyze_market", "explain_matches"],
    )
    graph.add_edge("analyze_market", "explain_matches")
    graph.add_edge("explain_matches", END)

    return graph
