"""LLM-powered search tools for roles, courses, and certifications."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from src.domain.job_matcher.schemas import (
    MarketDemand,
    MarketDemandResult,
    RoleReference,
    RoleSearchResult,
)
from src.domain.talent_forger.schemas import (
    CertSearchResult,
    CourseSearchResult,
    LearningResource,
)
from src.infrastructure.llm.deepseek_client import get_deepseek_llm

logger = logging.getLogger("uvicorn.error")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


async def search_job_roles(
    profile_summary: str,
    skills_text: str,
    recommended_roles: list[dict],
    detected_fields: list[dict],
    *,
    top_k: int = 5,
) -> list[RoleReference]:
    """Use LLM to generate relevant career role templates.

    Combines user profile data with double-diamond recommendations
    to produce realistic role templates with skill requirements.
    """
    prompt_template = _load_prompt("role_search.txt")

    roles_str = "\n".join(
        f"- {r.get('label', r.get('code', ''))}: {r.get('reason', '')}"
        for r in recommended_roles
    )
    fields_str = "\n".join(
        f"- {f.get('label', f.get('code', ''))}: score={f.get('score', 0)}"
        for f in detected_fields
    )

    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(RoleSearchResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "Profile: {profile_summary}\n\n"
            "Skills: {skills_text}\n\n"
            "Recommended Roles:\n{roles_str}\n\n"
            "Detected Fields:\n{fields_str}\n\n"
            "Generate {top_k} most relevant career roles with skill requirements.",
        ),
    ])

    chain = prompt | structured_llm
    result: RoleSearchResult = await chain.ainvoke({
        "profile_summary": profile_summary,
        "skills_text": skills_text,
        "roles_str": roles_str,
        "fields_str": fields_str,
        "top_k": str(top_k),
    })

    # Assign IDs if missing
    for role in result.roles:
        if not role.role_id:
            role.role_id = f"ROLE-{uuid.uuid4().hex[:8].upper()}"

    logger.info("[Search] Generated %d role templates", len(result.roles))
    return result.roles[:top_k]


async def search_courses(
    skill_name: str,
    current_level: str,
    required_level: str,
    *,
    min_results: int = 3,
) -> list[LearningResource]:
    """Use LLM to generate course recommendations for a skill gap."""
    prompt_template = _load_prompt("talent_forger_recommend.txt")

    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(CourseSearchResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "Find {min_results}+ courses for skill '{skill_name}' "
            "to go from {current_level} to {required_level} level.\n"
            "Include resource_type='Course' only. Provide real provider names "
            "and realistic URLs.",
        ),
    ])

    chain = prompt | structured_llm
    result: CourseSearchResult = await chain.ainvoke({
        "skill_name": skill_name,
        "current_level": current_level,
        "required_level": required_level,
        "min_results": str(min_results),
    })

    # Assign IDs
    for r in result.resources:
        if not r.resource_id:
            r.resource_id = f"RES-{uuid.uuid4().hex[:8].upper()}"
        r.skill_name = skill_name

    return result.resources


async def search_certifications(
    skill_name: str,
    current_level: str,
) -> list[LearningResource]:
    """Use LLM to generate certification recommendations for a skill."""
    prompt_template = _load_prompt("talent_forger_recommend.txt")

    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(CertSearchResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "Find relevant certifications for skill '{skill_name}' "
            "at {current_level} level.\n"
            "Include resource_type='Certification' only. Provide real "
            "certification bodies and realistic URLs.",
        ),
    ])

    chain = prompt | structured_llm
    result: CertSearchResult = await chain.ainvoke({
        "skill_name": skill_name,
        "current_level": current_level,
    })

    for r in result.resources:
        if not r.resource_id:
            r.resource_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"
        r.skill_name = skill_name

    return result.resources


async def analyze_market_demand(
    skills: list[str],
) -> list[MarketDemand]:
    """Use LLM to analyze market demand for a list of skills."""
    prompt_template = _load_prompt("market_demand.txt")

    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(MarketDemandResult)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "Analyze current job market demand for these skills:\n"
            "{skills_list}\n\n"
            "For each skill, provide a demand_score (0-1) and whether it is trending.",
        ),
    ])

    chain = prompt | structured_llm
    result: MarketDemandResult = await chain.ainvoke({
        "skills_list": "\n".join(f"- {s}" for s in skills),
    })

    return result.demands
