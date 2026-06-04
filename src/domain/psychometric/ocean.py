"""OCEAN (Big Five) scoring — BFI-10 instrument.

Design doc §7.1: 2 items per trait (one positive, one reverse-scored), 7-point
Likert scale.

    skor_trait       = mean(item_plus, 8 - item_reverse)   on 1..7  -> x(100/7) -> 0..100
    confidence_trait = 1 - |item_plus - (8 - item_reverse)| / 6
    confidence_OCEAN = mean(confidence_trait) x 100

Neuroticism is stored raw (never inverted) — fit against a role ideal is handled
downstream by the Matchmaker, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean

LIKERT_MIN = 1
LIKERT_MAX = 7
LIKERT_SPAN = LIKERT_MAX - LIKERT_MIN  # 6 — denominator for confidence
_REVERSE_PIVOT = LIKERT_MIN + LIKERT_MAX  # 8 — reverse(v) = 8 - v
_TO_PERCENT = 100 / LIKERT_MAX


class OceanTrait(str, Enum):
    OPENNESS = "O"
    CONSCIENTIOUSNESS = "C"
    EXTRAVERSION = "E"
    AGREEABLENESS = "A"
    NEUROTICISM = "N"


class Polarity(str, Enum):
    POSITIVE = "+"
    REVERSE = "-"


@dataclass(frozen=True)
class OceanResponse:
    trait: OceanTrait
    polarity: Polarity
    value: int  # 1..7


@dataclass(frozen=True)
class TraitScore:
    trait: OceanTrait
    score: float  # 0..100
    confidence: float  # 0..1


@dataclass(frozen=True)
class OceanResult:
    traits: dict[OceanTrait, TraitScore]
    confidence: float  # 0..100, mean per-trait confidence

    @property
    def scores(self) -> dict[str, float]:
        return {t.value: s.score for t, s in self.traits.items()}

    @property
    def trait_confidence(self) -> dict[str, float]:
        return {t.value: s.confidence for t, s in self.traits.items()}


def _validate_value(value: int) -> None:
    if not LIKERT_MIN <= value <= LIKERT_MAX:
        raise ValueError(
            f"OCEAN response value harus {LIKERT_MIN}-{LIKERT_MAX}, diterima: {value}"
        )


def score_ocean(responses: list[OceanResponse]) -> OceanResult:
    """Score a complete BFI-10 submission. Requires one '+' and one '-' per trait."""
    by_trait: dict[OceanTrait, dict[Polarity, int]] = {}
    for response in responses:
        _validate_value(response.value)
        polarities = by_trait.setdefault(response.trait, {})
        if response.polarity in polarities:
            raise ValueError(
                f"Duplikat item untuk trait {response.trait.value} "
                f"polarity {response.polarity.value}"
            )
        polarities[response.polarity] = response.value

    traits: dict[OceanTrait, TraitScore] = {}
    for trait in OceanTrait:
        polarities = by_trait.get(trait, {})
        if Polarity.POSITIVE not in polarities or Polarity.REVERSE not in polarities:
            raise ValueError(
                f"Trait {trait.value} butuh item '+' dan '-' (BFI-10 lengkap)"
            )

        item_plus = polarities[Polarity.POSITIVE]
        reverse_adjusted = _REVERSE_PIVOT - polarities[Polarity.REVERSE]
        raw = mean((item_plus, reverse_adjusted))  # 1..7

        traits[trait] = TraitScore(
            trait=trait,
            score=round(raw * _TO_PERCENT, 2),
            confidence=round(1 - abs(item_plus - reverse_adjusted) / LIKERT_SPAN, 4),
        )

    overall = mean(score.confidence for score in traits.values()) * 100
    return OceanResult(traits=traits, confidence=round(overall, 2))
