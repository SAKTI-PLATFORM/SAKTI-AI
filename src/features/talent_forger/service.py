"""TalentForger application service — orchestrates the LangGraph pipeline."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from src.domain.talent_forger.schemas import (
    LearningMaterial,
    LearningPath,
    LearningPathStep,
    LearningResource,
    ResourceRecommendation,
)
from src.domain.talent_forger.value_objects import TalentForgerOutput
from src.infrastructure.adapters.onboarding_adapter import parse_onboarding_data
from src.infrastructure.graphs.talent_forger_graph import build_talent_forger_graph
from src.infrastructure.observability.langfuse_client import get_langfuse_handler

logger = logging.getLogger("uvicorn.error")


class TalentForgerService:
    """Application service that compiles and runs the TalentForger graph."""

    def __init__(self) -> None:
        self._graph = build_talent_forger_graph()

    async def run(
        self,
        match_id: str,
        skill_gaps: list[dict[str, Any]],
        raw_input: dict[str, Any],
    ) -> TalentForgerOutput:
        """Execute the full TalentForger pipeline.

        Parameters
        ----------
        match_id : str
            The match ID from JobMatcher linking this learning path.
        skill_gaps : list[dict]
            Skill gap results from JobMatcher.
        raw_input : dict
            The full onboarding export JSON.

        Returns
        -------
        TalentForgerOutput
            Structured learning paths, steps, resources, and recommendations.
        """
        # 1. Parse raw input
        profile = parse_onboarding_data(raw_input)

        logger.info(
            "[TalentForger] Starting: user=%s match=%s gaps=%d",
            profile.user_id,
            match_id,
            len(skill_gaps),
        )

        # 2. Setup Langfuse tracing
        langfuse_handler = get_langfuse_handler(
            session_id=profile.onboarding_session_id,
            user_id=profile.user_id,
            module="TalentForger",
        )
        callbacks = [langfuse_handler] if langfuse_handler else []

        # 3. Compile and invoke graph
        compiled = self._graph.compile()

        initial_state = {
            "match_id": match_id,
            "user_profile": profile.model_dump(),
            "preferences": profile.preferences.model_dump() if profile.preferences else None,
            "skill_gaps": skill_gaps,
            "role_references": [],
            "course_candidates": [],
            "cert_candidates": [],
            "free_material_candidates": [],
            "learning_paths": [],
            "learning_path_steps": [],
            "resource_recommendations": [],
            "progress_step": 0,
            "trace_id": None,
        }

        config: RunnableConfig = {
            "callbacks": callbacks,
            "run_name": "TalentForger",
            "metadata": {
                "langfuse_session_id": profile.onboarding_session_id,
                "langfuse_user_id": profile.user_id,
                "langfuse_tags": ["TalentForger"],
                "langfuse_trace_name": "TalentForger_run"
            },
            "configurable": {"thread_id": f"{profile.onboarding_session_id}_{match_id}"},
        }

        from langchain_community.callbacks.manager import get_openai_callback

        # Run with streaming for progress updates
        final_state = None
        with get_openai_callback() as cb:
            async for chunk in compiled.astream(
                initial_state,
                config=config,
                stream_mode=("values", "custom"),
            ):
                if isinstance(chunk, tuple):
                    mode, data = chunk
                    if mode == "custom":
                        logger.info("[TalentForger] Progress: %s", data)
                    elif mode == "values":
                        final_state = data
                else:
                    final_state = chunk

            if final_state is None:
                # Fallback: invoke without streaming
                final_state = await compiled.ainvoke(initial_state, config=config)

        # DeepSeek V4 Flash pricing estimate (assuming cache miss for simplicity)
        deepseek_cost = (cb.prompt_tokens * 0.14 / 1_000_000) + (cb.completion_tokens * 0.28 / 1_000_000)
        
        logger.info(
            "[TalentForger] Token Usage: Total=%d (Prompt=%d, Completion=%d) Cost=$%.6f (DeepSeek V4 Flash est.)",
            cb.total_tokens,
            cb.prompt_tokens,
            cb.completion_tokens,
            deepseek_cost,
        )

        # 4. Extract and validate output
        learning_paths = [
            LearningPath(**lp) if isinstance(lp, dict) else lp
            for lp in final_state.get("learning_paths", [])
        ]
        learning_path_steps = [
            LearningPathStep(**s) if isinstance(s, dict) else s
            for s in final_state.get("learning_path_steps", [])
        ]
        all_resources = [
            LearningResource(**r) if isinstance(r, dict) else r
            for r in (
                final_state.get("course_candidates", [])
                + final_state.get("cert_candidates", [])
            )
        ]
        free_materials = [
            LearningMaterial(**m) if isinstance(m, dict) else m
            for m in final_state.get("free_material_candidates", [])
        ]
        recommendations = [
            ResourceRecommendation(**r) if isinstance(r, dict) else r
            for r in final_state.get("resource_recommendations", [])
        ]

        output = TalentForgerOutput(
            learning_paths=learning_paths,
            learning_path_steps=learning_path_steps,
            learning_resources=all_resources,
            free_materials=free_materials,
            resource_recommendations=recommendations,
        )

        logger.info(
            "[TalentForger] Complete: paths=%d steps=%d resources=%d free=%d recommendations=%d",
            len(output.learning_paths),
            len(output.learning_path_steps),
            len(output.learning_resources),
            len(output.free_materials),
            len(output.resource_recommendations),
        )

        return output
