"""RIASEC (Holland) scoring — 42-item binary worksheet.

Design doc §7.2: agree -> 1, disagree -> 0. Score per type = count of agreed items
in that type's column (0..7). Radar value = (score / 7) x 100. Holland code = the
three highest types (ties broken by canonical R-I-A-S-E-C order).

The item->type grid below is the source of truth; the client-supplied `letter` is
ignored on purpose so a mislabelled payload cannot corrupt scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

RIASEC_ITEMS_PER_TYPE = 7
RIASEC_TOTAL_ITEMS = 42
HOLLAND_CODE_LENGTH = 3


class RiasecType(str, Enum):
    REALISTIC = "R"
    INVESTIGATIVE = "I"
    ARTISTIC = "A"
    SOCIAL = "S"
    ENTERPRISING = "E"
    CONVENTIONAL = "C"


# Canonical column grid (design doc §7.2). Order of the dict defines tie-break order.
_COLUMNS: dict[RiasecType, tuple[int, ...]] = {
    RiasecType.REALISTIC: (1, 7, 14, 22, 30, 32, 37),
    RiasecType.INVESTIGATIVE: (2, 11, 18, 21, 26, 33, 39),
    RiasecType.ARTISTIC: (3, 8, 17, 23, 27, 31, 41),
    RiasecType.SOCIAL: (4, 12, 13, 20, 28, 34, 40),
    RiasecType.ENTERPRISING: (5, 10, 16, 19, 29, 36, 42),
    RiasecType.CONVENTIONAL: (6, 9, 15, 24, 25, 35, 38),
}

RIASEC_ORDER: tuple[RiasecType, ...] = tuple(_COLUMNS.keys())
ITEM_TO_TYPE: dict[int, RiasecType] = {
    item: riasec_type for riasec_type, items in _COLUMNS.items() for item in items
}


@dataclass(frozen=True)
class RiasecResponse:
    item: int  # 1..42
    agreed: bool


@dataclass(frozen=True)
class RiasecTypeScore:
    type: RiasecType
    raw: int  # 0..7
    score: float  # 0..100


@dataclass(frozen=True)
class RiasecResult:
    types: dict[RiasecType, RiasecTypeScore]
    holland_code: str  # three letters, e.g. "ICE"

    @property
    def scores(self) -> dict[str, float]:
        return {t.value: s.score for t, s in self.types.items()}

    @property
    def raw_scores(self) -> dict[str, int]:
        return {t.value: s.raw for t, s in self.types.items()}


def score_riasec(responses: list[RiasecResponse]) -> RiasecResult:
    """Score a complete 42-item RIASEC submission (every item exactly once)."""
    counts: dict[RiasecType, int] = {riasec_type: 0 for riasec_type in RiasecType}
    seen: set[int] = set()

    for response in responses:
        if not 1 <= response.item <= RIASEC_TOTAL_ITEMS:
            raise ValueError(
                f"Item RIASEC di luar rentang 1-{RIASEC_TOTAL_ITEMS}: {response.item}"
            )
        if response.item in seen:
            raise ValueError(f"Item RIASEC duplikat: {response.item}")
        seen.add(response.item)
        if response.agreed:
            counts[ITEM_TO_TYPE[response.item]] += 1

    if len(seen) != RIASEC_TOTAL_ITEMS:
        missing = sorted(set(ITEM_TO_TYPE) - seen)
        raise ValueError(f"Jawaban RIASEC tidak lengkap, item hilang: {missing}")

    types = {
        riasec_type: RiasecTypeScore(
            type=riasec_type,
            raw=count,
            score=round(count / RIASEC_ITEMS_PER_TYPE * 100, 2),
        )
        for riasec_type, count in counts.items()
    }

    ranked = sorted(
        RiasecType,
        key=lambda t: (-counts[t], RIASEC_ORDER.index(t)),
    )
    holland_code = "".join(t.value for t in ranked[:HOLLAND_CODE_LENGTH])

    return RiasecResult(types=types, holland_code=holland_code)
