"""
RAG-B — AI Insight (grounded).

Pure domain: turns candidate facts into a "Market Ready" narrative. Numbers are
NEVER invented — every figure in the output comes from the facts passed in
(employability, match score, gap hours). This deterministic narrative also doubles
as the grounding context / fallback when no LLM key is configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Employability threshold above which the candidate is framed "Market Ready".
MARKET_READY_THRESHOLD = 70.0


@dataclass(frozen=True)
class GapFact:
    skill: str
    gap_hours: int
    priority: str


@dataclass(frozen=True)
class InsightFacts:
    full_name: str
    target_role: str
    employability_score: float
    profile_completeness: float
    matched_count: int = 0
    top_match_title: str | None = None
    top_match_score: float | None = None  # 0..1
    strengths: list[str] = field(default_factory=list)
    top_gaps: list[GapFact] = field(default_factory=list)
    holland_code: str | None = None

    @property
    def market_ready(self) -> bool:
        return self.employability_score >= MARKET_READY_THRESHOLD


def _first_name(full_name: str) -> str:
    return full_name.strip().split(" ")[0] if full_name.strip() else "Kandidat"


def build_grounded_insight(facts: InsightFacts) -> str:
    """Deterministic, fully-grounded Indonesian narrative for the AI Insight card."""
    name = _first_name(facts.full_name)
    status = "Market Ready" if facts.market_ready else "Hampir Siap"
    score = round(facts.employability_score)

    parts: list[str] = [
        f"Status {name} saat ini adalah “{status}” untuk peran "
        f"{facts.target_role} dengan Employability Score {score}%."
    ]

    if facts.strengths:
        top = ", ".join(facts.strengths[:3])
        parts.append(f"Kekuatan utama Anda ada di {top}.")

    if facts.top_match_title and facts.top_match_score is not None:
        parts.append(
            f"Posisi paling cocok saat ini adalah {facts.top_match_title} "
            f"dengan match score {round(facts.top_match_score * 100)}%."
        )

    if facts.top_gaps:
        gap = facts.top_gaps[0]
        parts.append(
            f"Untuk naik level, fokus pada penguatan {gap.skill} "
            f"(prioritas {gap.priority}, perlu sekitar {gap.gap_hours} jam latihan)."
        )

    if not facts.market_ready:
        parts.append(
            "Lengkapi gap skill prioritas tinggi untuk melewati ambang "
            "Market Ready dan mengaktifkan AutoApply."
        )

    return " ".join(parts)


def build_grounding_prompt(facts: InsightFacts) -> str:
    """Prompt for an LLM — instructs it to stay strictly grounded in the facts."""
    gap_lines = "\n".join(
        f"- {g.skill}: butuh {g.gap_hours} jam (prioritas {g.priority})"
        for g in facts.top_gaps
    ) or "- (tidak ada gap signifikan)"
    return (
        "Anda adalah career advisor SAKTI AI. Tulis 2-3 kalimat narasi 'AI Insight' "
        "dalam Bahasa Indonesia yang memotivasi namun realistis. ATURAN KETAT: "
        "jangan mengarang angka apa pun; gunakan HANYA fakta di bawah.\n\n"
        f"Nama: {facts.full_name}\n"
        f"Target role: {facts.target_role}\n"
        f"Employability score: {round(facts.employability_score)}%\n"
        f"Market ready: {'ya' if facts.market_ready else 'belum'}\n"
        f"Kekuatan: {', '.join(facts.strengths) or '-'}\n"
        f"Top match: {facts.top_match_title or '-'} "
        f"({round((facts.top_match_score or 0) * 100)}%)\n"
        f"Skill gap:\n{gap_lines}\n"
    )
