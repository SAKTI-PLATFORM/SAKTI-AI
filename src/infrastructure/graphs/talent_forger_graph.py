"""TalentForger LangGraph — learning path generation pipeline.

Graph flow:
    START → get_job_references → get_course_references
                               → get_material_references  (parallel)
                               → get_cert_references
          → build_learning_path → explain_recommendations → END
"""

from __future__ import annotations

import logging
import uuid
from typing import cast
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.core.config import settings
from src.domain.job_matcher.schemas import RoleReference, SkillGapResult
from src.domain.talent_forger.schemas import (
    LearningPath,
    LearningPathPlan,
    LearningPathStep,
    LearningResource,
    ResourceRecommendation,
)
from src.domain.talent_forger.value_objects import TalentForgerState
from src.infrastructure.llm.deepseek_client import get_deepseek_llm
from src.infrastructure.tools.search import (
    search_certifications,
    search_courses,
    search_learning_materials,
)

logger = logging.getLogger("uvicorn.error")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


# ── Node functions ──────────────────────────────────────


async def get_job_references(state: TalentForgerState) -> dict:
    """Load role references from the matched career results."""
    writer = get_stream_writer()
    writer({"step": 1, "total": 5, "title": "Memuat referensi role..."})

    # The role references come from the JobMatcher results passed via state
    # For now, construct minimal references from skill_gaps match data
    role_refs = []
    seen_matches = set()

    # Limit skill gaps for Lite users (e.g. only process top 3 high priority gaps)
    skill_gaps = state.get("skill_gaps", [])
    preferences = state.get("preferences")
    if isinstance(preferences, dict):
        is_lite = preferences.get("subscription_tier", "Pro") == "Lite"
    else:
        is_lite = getattr(preferences, "subscription_tier", "Pro") == "Lite" if preferences else False

    if is_lite:
        # Sort gaps by priority (High > Medium > Low) and limit to top 3
        priority_map = {"High": 0, "Medium": 1, "Low": 2}
        skill_gaps = sorted(
            skill_gaps, 
            key=lambda g: priority_map.get(g.get("priority", "Low") if isinstance(g, dict) else getattr(g, "priority", "Low"), 3)
        )[:3]
        logger.info("[TalentForger] Lite mode: limited skill gaps to %d", len(skill_gaps))

    for gap_dict in skill_gaps:
        match_id = gap_dict.get("match_id", "") if isinstance(gap_dict, dict) else getattr(gap_dict, "match_id", "")
        if match_id not in seen_matches:
            seen_matches.add(match_id)
            role_refs.append({
                "role_id": f"REF-{match_id}",
                "role_name": "Target Role",
                "role_category": "General",
                "role_level": "Mid",
                "description": f"Role associated with match {match_id}",
            })

    return {
        "role_references": role_refs,
        "skill_gaps": skill_gaps,
        "progress_step": 1,
    }


async def get_course_references(state: TalentForgerState) -> dict:
    """Search for course resources for each skill gap."""
    writer = get_stream_writer()
    writer({"step": 2, "total": 5, "title": "Mencari kursus yang relevan..."})

    courses = []
    for gap_dict in state.get("skill_gaps", []):
        gap = SkillGapResult(**gap_dict) if isinstance(gap_dict, dict) else gap_dict

        try:
            gap_courses = await search_courses(
                skill_name=gap.skill_name,
                current_level=gap.current_level,
                required_level=gap.required_level,
                min_results=settings.min_resources_per_gap,
            )
            courses.extend([c.model_dump() for c in gap_courses])
        except Exception:
            logger.exception(
                "[TalentForger] Course search failed for skill=%s",
                gap.skill_name,
            )

    return {
        "course_candidates": courses,
    }


async def get_cert_references(state: TalentForgerState) -> dict:
    """Search for certification resources for skill gaps."""
    writer = get_stream_writer()
    writer({"step": 3, "total": 5, "title": "Mencari sertifikasi yang relevan..."})

    certs = []
    seen_skills = set()

    for gap_dict in state.get("skill_gaps", []):
        gap = SkillGapResult(**gap_dict) if isinstance(gap_dict, dict) else gap_dict

        # Avoid duplicate searches for the same skill
        if gap.skill_name.lower() in seen_skills:
            continue
        seen_skills.add(gap.skill_name.lower())

        try:
            gap_certs = await search_certifications(
                skill_name=gap.skill_name,
                current_level=gap.current_level,
            )
            certs.extend([c.model_dump() for c in gap_certs])
        except Exception:
            logger.exception(
                "[TalentForger] Cert search failed for skill=%s",
                gap.skill_name,
            )

    return {
        "cert_candidates": certs,
    }


async def get_material_references(state: TalentForgerState) -> dict:
    """Search for free learning materials (articles, videos, docs, GitHub).

    Runs in parallel with get_cert_references in the pipeline.
    Focuses on free, high-quality content including Indonesian resources.
    """
    writer = get_stream_writer()
    writer({"step": 4, "total": 6, "title": "Mencari materi belajar gratis..."})

    materials = []
    seen_skills: set[str] = set()

    for gap_dict in state.get("skill_gaps", []):
        gap = SkillGapResult(**gap_dict) if isinstance(gap_dict, dict) else gap_dict

        if gap.skill_name.lower() in seen_skills:
            continue
        seen_skills.add(gap.skill_name.lower())

        try:
            gap_materials = await search_learning_materials(
                skill_name=gap.skill_name,
                current_level=gap.current_level,
                min_results=3,
                prefer_indonesian=True,
            )
            materials.extend([m.model_dump() for m in gap_materials])
        except Exception:
            logger.exception(
                "[TalentForger] Material search failed for skill=%s",
                gap.skill_name,
            )

    return {
        "free_material_candidates": materials,
    }


async def build_learning_path(state: TalentForgerState) -> dict:
    """Use LLM to build a structured learning path from skill gaps and resources."""
    writer = get_stream_writer()
    writer({"step": 4, "total": 5, "title": "Menyusun jalur pembelajaran..."})

    import json

    match_id = state.get("match_id", "UNKNOWN")
    skill_gaps = state.get("skill_gaps", [])
    # Include all resource types: paid courses, certs, and free materials
    all_resources = (
        state.get("course_candidates", [])
        + state.get("cert_candidates", [])
        + state.get("free_material_candidates", [])
    )

    # Build the learning path via LLM
    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(LearningPathPlan, method="json_mode")

    preferences = state.get("preferences")
    if isinstance(preferences, dict):
        is_lite = preferences.get("subscription_tier", "Pro") == "Lite"
    else:
        is_lite = getattr(preferences, "subscription_tier", "Pro") == "Lite" if preferences else False
    
    rules = (
        "1. Order steps by gap priority (High → Medium → Low)\n"
        "2. Group related skills in the same week when possible\n"
        "3. Assign realistic week numbers (1-12)\n"
        "4. Each step should reference a specific skill gap\n"
        "5. Create resource recommendations linking steps to available resources\n"
        "6. Use IDs that follow the pattern: LP-xxx, STEP-xxx, REC-xxx\n\n"
    )
    
    if is_lite:
        rules += "7. LITE VERSION RESTRICTION: Keep the learning path short (maximum 3 weeks/steps). For a full roadmap, the user must upgrade to Pro.\n\n"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a learning path architect for the SAKTI platform. "
            "Create a structured, sequential learning path that addresses "
            "skill gaps in priority order. Each step should have a clear "
            "objective and be linked to specific resources.\n\n"
            "Rules:\n"
            f"{rules}"
            "Output must be a JSON object containing exact keys: 'learning_path', 'steps' (array), and 'recommendations' (array).\n\n"
            "Format your output EXACTLY like this JSON structure:\n"
            "{{\n"
            '  "learning_path": {{"target_role": "...", "learning_path_type": "structured", "estimated_duration_weeks": 4}},\n'
            '  "steps": [{{"step_id": "STEP-001", "step_order": 1, "week": 1, "topic": "...", "objective": "...", "gap_id": "...", "related_skill_name": "..."}}],\n'
            '  "recommendations": [{{"recommendation_id": "REC-001", "step_id": "STEP-001", "resource_id": "RES-001", "recommendation_reason": "...", "priority_order": 1}}]\n'
            "}}",
        ),
        (
            "human",
            "Match ID: {match_id}\n\n"
            "Skill Gaps:\n{gaps_json}\n\n"
            "Available Resources:\n{resources_json}\n\n"
            "Build a comprehensive learning path.",
        ),
    ])

    try:
        chain = prompt | structured_llm
        result = await chain.ainvoke({
            "match_id": match_id,
            "gaps_json": json.dumps(skill_gaps, default=str),
            "resources_json": json.dumps(all_resources[:30], default=str),
        })
        plan = cast(LearningPathPlan, result)

        lp_id = plan.learning_path.learning_path_id
        if not lp_id:
            lp_id = f"LP-{uuid.uuid4().hex[:8].upper()}"
            plan.learning_path.learning_path_id = lp_id
        plan.learning_path.match_id = match_id

        # Make sure steps link back to learning path
        for step in plan.steps:
            step.learning_path_id = lp_id

        return {
            "learning_paths": [plan.learning_path.model_dump()],
            "learning_path_steps": [s.model_dump() for s in plan.steps],
            "resource_recommendations": [r.model_dump() for r in plan.recommendations],
            "progress_step": 4,
        }
    except Exception:
        logger.exception("[TalentForger] Failed to build learning path")

        # Fallback: create a simple learning path
        lp_id = f"LP-{uuid.uuid4().hex[:8].upper()}"
        fallback_path = LearningPath(
            learning_path_id=lp_id,
            match_id=match_id,
            target_role="Target Role",
            learning_path_type="auto-generated",
            estimated_duration_weeks=max(len(skill_gaps) * 2, 4),
        )

        steps = []
        recommendations = []
        for i, gap_dict in enumerate(skill_gaps):
            step_id = f"STEP-{uuid.uuid4().hex[:8].upper()}"
            step = LearningPathStep(
                step_id=step_id,
                learning_path_id=lp_id,
                gap_id=gap_dict.get("gap_id", ""),
                step_order=i + 1,
                week=i + 1,
                topic=gap_dict.get("skill_name", ""),
                objective=f"Improve {gap_dict.get('skill_name', '')} "
                         f"from {gap_dict.get('current_level', 'None')} "
                         f"to {gap_dict.get('required_level', 'Beginner')}",
                related_skill_name=gap_dict.get("skill_name", ""),
            )
            steps.append(step)

            # Link available resources
            skill_resources = [
                r for r in all_resources
                if r.get("skill_name", "").lower() == gap_dict.get("skill_name", "").lower()
            ]
            for j, res in enumerate(skill_resources[:3]):
                rec = ResourceRecommendation(
                    recommendation_id=f"REC-{uuid.uuid4().hex[:8].upper()}",
                    step_id=step_id,
                    resource_id=res.get("resource_id", ""),
                    recommendation_reason=f"Resource for {gap_dict.get('skill_name', '')} skill gap",
                    priority_order=j + 1,
                )
                recommendations.append(rec)

        return {
            "learning_paths": [fallback_path.model_dump()],
            "learning_path_steps": [s.model_dump() for s in steps],
            "resource_recommendations": [r.model_dump() for r in recommendations],
            "progress_step": 4,
        }


async def explain_recommendations(state: TalentForgerState) -> dict:
    """Generate human-readable explanations for resource recommendations."""
    writer = get_stream_writer()
    writer({"step": 5, "total": 5, "title": "Menyusun rekomendasi akhir..."})

    # The recommendations already have reasons from the learning path builder
    # This node enriches them with contextual explanations if needed

    import json

    recommendations = state.get("resource_recommendations", [])
    if not recommendations:
        return {"progress_step": 5}

    profile = state["user_profile"]
    if isinstance(profile, dict):
        learning_style = (profile.get("preferences") or {}).get("learning_style")
    else:
        learning_style = profile.preferences.learning_style if profile.preferences else None

    llm = get_deepseek_llm()

    # Batch-enrich recommendations with personalized reasons
    all_resources = (
        state.get("course_candidates", [])
        + state.get("cert_candidates", [])
        + state.get("free_material_candidates", [])
    )
    resource_map = {r.get("resource_id", ""): r for r in all_resources}

    for rec in recommendations:
        if isinstance(rec, dict):
            resource = resource_map.get(rec.get("resource_id", ""))
            if resource and not rec.get("recommendation_reason"):
                rec["recommendation_reason"] = (
                    f"Direkomendasikan untuk meningkatkan skill "
                    f"{resource.get('skill_name', '')} melalui "
                    f"{resource.get('resource_type', 'resource')} "
                    f"dari {resource.get('provider', 'provider')}."
                )

    return {"progress_step": 5}


# ── Graph builder ──────────────────────────────────────


def build_talent_forger_graph() -> StateGraph:
    """Build and return the compiled TalentForger LangGraph."""
    graph = StateGraph(TalentForgerState)  # type: ignore

    # Add nodes
    graph.add_node("get_job_references", get_job_references)
    graph.add_node("get_course_references", get_course_references)
    graph.add_node("get_cert_references", get_cert_references)
    graph.add_node("get_material_references", get_material_references)
    graph.add_node("build_learning_path", build_learning_path)
    graph.add_node("explain_recommendations", explain_recommendations)

    # Wire edges
    # get_job_references fans out to 3 parallel nodes
    graph.add_edge(START, "get_job_references")
    graph.add_edge("get_job_references", "get_course_references")
    graph.add_edge("get_job_references", "get_cert_references")
    graph.add_edge("get_job_references", "get_material_references")
    # All 3 parallel nodes fan back in to build_learning_path
    graph.add_edge("get_course_references", "build_learning_path")
    graph.add_edge("get_cert_references", "build_learning_path")
    graph.add_edge("get_material_references", "build_learning_path")
    graph.add_edge("build_learning_path", "explain_recommendations")
    graph.add_edge("explain_recommendations", END)

    return graph
