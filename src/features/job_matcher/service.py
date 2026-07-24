"""JobMatcher application service — orchestrates the LangGraph pipeline."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from src.domain.job_matcher.schemas import (
    CareerMatchResult,
    ScoreDetail,
    SkillGapResult,
)
from src.domain.job_matcher.value_objects import JobMatcherOutput
from src.infrastructure.adapters.onboarding_adapter import parse_onboarding_data
from src.infrastructure.graphs.job_matcher_graph import build_job_matcher_graph
from src.infrastructure.observability.langfuse_client import get_langfuse_handler

logger = logging.getLogger("uvicorn.error")


class JobMatcherService:
    """Application service that compiles and runs the JobMatcher graph."""

    def __init__(self) -> None:
        self._graph = build_job_matcher_graph()

    async def run(self, raw_input: dict[str, Any]) -> JobMatcherOutput:
        """Execute the full JobMatcher pipeline.

        Parameters
        ----------
        raw_input : dict
            The full onboarding export JSON (same shape as
            ``result-full_AFTER_SKILL_PARSERS.json``).

        Returns
        -------
        JobMatcherOutput
            Structured career match results with score breakdowns and skill gaps.
        """
        # 1. Parse raw input into domain model
        profile = parse_onboarding_data(raw_input)

        logger.info(
            "[JobMatcher] Starting: user=%s skills=%d session=%s",
            profile.user_id,
            len(profile.skills),
            profile.onboarding_session_id,
        )

        # 2. Setup Langfuse tracing
        langfuse_handler = get_langfuse_handler(
            session_id=profile.onboarding_session_id,
            user_id=profile.user_id,
            module="JobMatcher",
        )
        callbacks = [langfuse_handler] if langfuse_handler else []

        # 3. Compile and invoke graph
        compiled = self._graph.compile()

        initial_state = {
            "user_profile": profile.model_dump(),
            "preferences": profile.preferences.model_dump() if profile.preferences else None,
            "candidate_roles": [],
            "similarity_results": [],
            "score_details": [],
            "career_matches": [],
            "skill_gaps": [],
            "market_demand": {},
            "progress_step": 0,
            "trace_id": None,
        }

        config: RunnableConfig = {
            "callbacks": callbacks,
            "configurable": {"thread_id": profile.onboarding_session_id},
        }

        # Run with streaming for progress updates
        final_state = None
        async for chunk in compiled.astream(
            initial_state,
            config=config,
            stream_mode=["updates", "custom"],
        ):
            if isinstance(chunk, tuple):
                mode, data = chunk
                if mode == "custom":
                    logger.info("[JobMatcher] Progress: %s", data)
                elif mode == "updates":
                    final_state = data
            else:
                final_state = chunk

        if final_state is None:
            # Fallback: invoke without streaming
            final_state = await compiled.ainvoke(initial_state, config=config)

        # 4. Extract and validate output
        career_matches = [
            CareerMatchResult(**m) if isinstance(m, dict) else m
            for m in final_state.get("career_matches", [])
        ]
        score_details = [
            ScoreDetail(**s) if isinstance(s, dict) else s
            for s in final_state.get("score_details", [])
        ]
        skill_gaps = [
            SkillGapResult(**g) if isinstance(g, dict) else g
            for g in final_state.get("skill_gaps", [])
        ]

        # Sort by total_match_score descending
        career_matches.sort(key=lambda m: m.total_match_score, reverse=True)

        output = JobMatcherOutput(
            career_match_results=career_matches,
            career_match_score_details=score_details,
            skill_gap_results=skill_gaps,
        )

        logger.info(
            "[JobMatcher] Complete: matches=%d gaps=%d top_score=%.1f",
            len(output.career_match_results),
            len(output.skill_gap_results),
            output.career_match_results[0].total_match_score
            if output.career_match_results
            else 0,
        )

        return output
