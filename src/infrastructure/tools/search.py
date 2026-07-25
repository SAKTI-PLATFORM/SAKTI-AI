"""LLM-powered search tools for roles, courses, and certifications.

Architecture (Retrieve → Augment → Generate pattern):
    1. RETRIEVE  — Brave LLM Context API fetches real web data
    2. AUGMENT   — Grounding snippets are injected into the LLM prompt
    3. GENERATE  — DeepSeek generates structured output, grounded in real data

If BRAVE_API_KEY is not set, tools fall back to pure LLM generation.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import cast

from langchain_core.prompts import ChatPromptTemplate

from src.core.config import settings
from src.domain.job_matcher.schemas import (
    MarketDemand,
    MarketDemandResult,
    RoleReference,
    RoleSearchResult,
)
from src.domain.talent_forger.schemas import (
    CertSearchResult,
    CourseSearchResult,
    LearningMaterial,
    LearningMaterialResult,
    LearningResource,
)
from src.infrastructure.llm.deepseek_client import get_deepseek_llm
from src.infrastructure.search.brave_search_client import get_brave_client

logger = logging.getLogger("uvicorn.error")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


async def _brave_search(query: str, *, freshness: str = "", max_items: int = 5) -> str:
    """Fetch real web context from Brave and return a formatted string.

    Returns an empty string if Brave is not configured so callers can
    handle the no-search fallback cleanly.
    """
    if not settings.brave_search_enabled:
        return ""

    try:
        client = get_brave_client()
        result = await client.search(
            query,
            count=15,
            maximum_number_of_tokens=settings.brave_max_tokens,
            maximum_number_of_urls=settings.brave_max_urls,
            maximum_number_of_snippets=25,
            freshness=freshness,
            context_threshold_mode="balanced",
        )
        context = result.as_combined_context(max_items=max_items)
        logger.debug(
            "[BraveSearch] Got %d items for: %r (sources: %s)",
            len(result.items),
            query,
            ", ".join(result.source_urls[:3]),
        )
        return context
    except Exception:
        logger.exception("[BraveSearch] Search failed for: %r", query)
        return ""


def _brave_grounding_section(context: str) -> str:
    """Wrap Brave context in a clear LLM-readable section."""
    if not context:
        return ""
    return (
        "\n\n## Data Riil dari Web (Brave Search)\n"
        "Gunakan informasi berikut sebagai referensi faktual untuk output kamu:\n\n"
        f"{context}\n"
        "---\n"
    )


# ── Job Roles Search ───────────────────────────────────


async def search_job_roles(
    profile_summary: str,
    skills_text: str,
    recommended_roles: list[dict],
    detected_fields: list[dict],
    *,
    top_k: int = 5,
) -> RoleSearchResult:
    """Search for relevant career roles, grounded with real job market data.

    Flow:
        1. Query Brave for current job postings in Indonesia matching the profile.
        2. Inject real results into the LLM prompt for grounded generation.
        3. DeepSeek generates typed RoleReference objects.
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

    # ── RETRIEVE: Brave web search ─────────────────────
    # Build a query from the top detected field and skills
    top_field = detected_fields[0].get("label", "") if detected_fields else ""
    top_skills = skills_text.split(",")[:3]
    brave_query = (
        f"lowongan kerja {top_field} {' '.join(top_skills)} "
        f"Indonesia 2025 skill requirements"
    ).strip()
    brave_context = await _brave_search(brave_query, freshness="pm", max_items=4)
    grounding_section = _brave_grounding_section(brave_context)

    # ── AUGMENT + GENERATE ─────────────────────────────
    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(RoleSearchResult, method="json_mode")

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "Profile: {profile_summary}\n\n"
            "Skills: {skills_text}\n\n"
            "Recommended Roles:\n{roles_str}\n\n"
            "Detected Fields:\n{fields_str}\n\n"
            "{grounding_section}"
            "Hasilkan {top_k} role karir yang paling relevan dengan persyaratan skill-nya.",
        ),
    ])

    chain = prompt | structured_llm
    result = cast(RoleSearchResult, await chain.ainvoke({
        "profile_summary": profile_summary,
        "skills_text": skills_text,
        "roles_str": roles_str,
        "fields_str": fields_str,
        "grounding_section": grounding_section,
        "top_k": str(top_k),
    }))

    for role in result.roles:
        if not role.role_id:
            role.role_id = f"ROLE-{uuid.uuid4().hex[:8].upper()}"

    logger.info("[Search] Generated %d role templates and %d active job postings", len(result.roles), len(result.active_job_postings))
    
    # Trim to top_k if needed
    result.roles = result.roles[:top_k]
    return result


# ── Course Search ──────────────────────────────────────


async def search_courses(
    skill_name: str,
    current_level: str,
    required_level: str,
    *,
    min_results: int = 3,
) -> list[LearningResource]:
    """Search for real courses, grounded with Brave web data.

    Flow:
        1. Query Brave for actual course listings for the skill gap.
        2. Inject real course data into LLM prompt.
        3. DeepSeek returns structured LearningResource objects.
    """
    prompt_template = _load_prompt("talent_forger_recommend.txt")

    # ── RETRIEVE ───────────────────────────────────────
    brave_query = (
        f"kursus online {skill_name} {required_level} "
        f"Coursera Udemy Dicoding site:coursera.org OR site:udemy.com OR site:dicoding.com"
    )
    brave_context = await _brave_search(brave_query, max_items=4)
    grounding_section = _brave_grounding_section(brave_context)

    # ── AUGMENT + GENERATE ─────────────────────────────
    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(CourseSearchResult, method="json_mode")

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "{grounding_section}"
            "Cari {min_results}+ kursus untuk skill '{skill_name}' "
            "dari level {current_level} ke {required_level}.\n"
            "resource_type harus 'Course'. "
            "Gunakan kursus nyata dari data web di atas jika tersedia, "
            "atau rekomendasikan kursus terkenal di platform ternama.",
        ),
    ])

    chain = prompt | structured_llm
    result = cast(CourseSearchResult, await chain.ainvoke({
        "skill_name": skill_name,
        "current_level": current_level,
        "required_level": required_level,
        "min_results": str(min_results),
        "grounding_section": grounding_section,
    }))

    for r in result.resources:
        if not r.resource_id:
            r.resource_id = f"RES-{uuid.uuid4().hex[:8].upper()}"
        r.skill_name = skill_name

    return result.resources


# ── Certification Search ───────────────────────────────


async def search_certifications(
    skill_name: str,
    current_level: str,
) -> list[LearningResource]:
    """Search for relevant certifications, grounded with Brave web data."""
    prompt_template = _load_prompt("talent_forger_recommend.txt")

    # ── RETRIEVE ───────────────────────────────────────
    brave_query = (
        f"sertifikasi {skill_name} profesional terbaik 2025 "
        f"site:aws.amazon.com OR site:google.com OR site:microsoft.com OR site:credly.com"
    )
    brave_context = await _brave_search(brave_query, max_items=3)
    grounding_section = _brave_grounding_section(brave_context)

    # ── AUGMENT + GENERATE ─────────────────────────────
    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(CertSearchResult, method="json_mode")

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "{grounding_section}"
            "Cari sertifikasi profesional yang relevan untuk skill '{skill_name}' "
            "di level {current_level}.\n"
            "resource_type harus 'Certification'. "
            "Utamakan sertifikasi dari badan resmi seperti AWS, Google Cloud, "
            "Microsoft, PMI, atau badan sertifikasi lain yang diakui industri.",
        ),
    ])

    chain = prompt | structured_llm
    result = cast(CertSearchResult, await chain.ainvoke({
        "skill_name": skill_name,
        "current_level": current_level,
        "grounding_section": grounding_section,
    }))

    for r in result.resources:
        if not r.resource_id:
            r.resource_id = f"CERT-{uuid.uuid4().hex[:8].upper()}"
        r.skill_name = skill_name

    return result.resources


# ── Market Demand Analysis ─────────────────────────────


async def analyze_market_demand(
    skills: list[str],
) -> list[MarketDemand]:
    """Analyze job market demand, grounded with real Brave job market data.

    Flow:
        1. Query Brave for current Indonesian tech job market trends.
        2. Inject data into the LLM prompt.
        3. DeepSeek produces evidence-based demand scores.
    """
    prompt_template = _load_prompt("market_demand.txt")

    # ── RETRIEVE ───────────────────────────────────────
    skills_query = " ".join(skills[:5])
    brave_query = (
        f"tren permintaan skill {skills_query} pasar kerja teknologi "
        f"Indonesia 2025 lowongan LinkedIn Glints Jobstreet"
    )
    brave_context = await _brave_search(
        brave_query,
        freshness="pm",   # Last 31 days — fresh market data
        max_items=4,
    )
    grounding_section = _brave_grounding_section(brave_context)

    # ── AUGMENT + GENERATE ─────────────────────────────
    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(MarketDemandResult, method="json_mode")

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "{grounding_section}"
            "Analisis tingkat permintaan pasar kerja saat ini untuk skill-skill berikut:\n"
            "{skills_list}\n\n"
            "Untuk setiap skill, berikan demand_score (0-1) dan status trending. "
            "Gunakan data dari web di atas sebagai acuan faktual.",
        ),
    ])

    chain = prompt | structured_llm
    result = cast(MarketDemandResult, await chain.ainvoke({
        "skills_list": "\n".join(f"- {s}" for s in skills),
        "grounding_section": grounding_section,
    }))

    return result.demands


# ── Free Learning Materials Search ──────────────────────


async def search_learning_materials(
    skill_name: str,
    current_level: str,
    *,
    min_results: int = 4,
    prefer_indonesian: bool = True,
) -> list[LearningMaterial]:
    """Search for FREE learning materials: docs, videos, articles, GitHub repos.

    Covers sources like official docs, YouTube tutorials, freeCodeCamp,
    Medium, roadmap.sh, GitHub awesome-lists, and Indonesian platforms
    like Dicoding and Petani Kode.

    Flow:
        1. RETRIEVE  — Brave searches for free tutorials + content for the skill
        2. AUGMENT   — Real URLs and titles injected into LLM prompt
        3. GENERATE  — DeepSeek curates and structures the best materials

    Parameters
    ----------
    skill_name : str
        The skill to search materials for.
    current_level : str
        User's current proficiency level.
    min_results : int
        Minimum number of materials to return.
    prefer_indonesian : bool
        If True, adds Indonesian-language search terms to the Brave query.
    """
    prompt_template = _load_prompt("learning_materials.txt")

    # ── RETRIEVE: Multi-angle search for free content ─────
    indonesian_hint = "bahasa indonesia tutorial" if prefer_indonesian else ""

    # Search 1: General free tutorials
    query_tutorial = (
        f"tutorial {skill_name} gratis {current_level} "
        f"{indonesian_hint} site:freecodecamp.org OR site:youtube.com OR site:medium.com"
    ).strip()

    # Search 2: Official docs + GitHub
    query_docs = (
        f"{skill_name} documentation tutorial beginner github "
        f"site:github.com OR site:roadmap.sh OR site:developer.mozilla.org"
    )

    # Search 3: Indonesian platforms
    query_id = (
        f"belajar {skill_name} tutorial gratis dicoding petanikode"
    ) if prefer_indonesian else ""

    # Run searches concurrently for speed
    import asyncio
    tasks = [
        _brave_search(query_tutorial, max_items=3),
        _brave_search(query_docs, max_items=2),
    ]
    if query_id:
        tasks.append(_brave_search(query_id, max_items=2))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge non-empty context blocks
    combined_context = ""
    for r in results:
        if isinstance(r, str) and r:
            combined_context += r + "\n"

    grounding_section = _brave_grounding_section(combined_context)

    # ── AUGMENT + GENERATE ──────────────────────────
    llm = get_deepseek_llm()
    structured_llm = llm.with_structured_output(LearningMaterialResult, method="json_mode")

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        (
            "human",
            "{grounding_section}"
            "Kurasi {min_results}+ materi belajar GRATIS untuk skill '{skill_name}' "
            "di level {current_level}.\n\n"
            "Sertakan campuran tipe konten: Article, Video, Documentation, GitHub, Tutorial.\n"
            "{indonesian_note}"
            "Gunakan URL nyata dari data web di atas jika tersedia.",
        ),
    ])

    indonesian_note = (
        "Prioritaskan minimal 1-2 konten berbahasa Indonesia jika berkualitas baik.\n"
        if prefer_indonesian else ""
    )

    chain = prompt | structured_llm
    result = cast(LearningMaterialResult, await chain.ainvoke({
        "skill_name": skill_name,
        "current_level": current_level,
        "min_results": str(min_results),
        "grounding_section": grounding_section,
        "indonesian_note": indonesian_note,
    }))

    # Assign IDs and enforce skill_name
    for i, mat in enumerate(result.materials):
        if not mat.resource_id:
            mat.resource_id = f"MAT-{uuid.uuid4().hex[:8].upper()}"
        if not mat.skill_name:
            mat.skill_name = skill_name

    logger.debug(
        "[Search] Found %d learning materials for '%s' (level=%s)",
        len(result.materials),
        skill_name,
        current_level,
    )
    return result.materials
