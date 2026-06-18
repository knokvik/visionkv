"""Shared policy objects for VisionKV decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .pytorch_prototype import (
    PrefetchBudgetRecommendation,
    TransferReport,
    recommend_prefetch_block_count,
)


@dataclass(frozen=True)
class VisionKVPolicy:
    """Shared knobs that connect prototype findings to integration behavior."""

    hot_prefetch_block_count: int | None = None
    flashback_budget_ms: float = 50.0
    background_prefetch_remainder: bool = True

    @classmethod
    def from_prefetch_budget_recommendation(
        cls,
        recommendation: PrefetchBudgetRecommendation,
        background_prefetch_remainder: bool = True,
    ) -> "VisionKVPolicy":
        return cls(
            hot_prefetch_block_count=recommendation.recommended_block_count,
            flashback_budget_ms=recommendation.latency_budget_ms,
            background_prefetch_remainder=background_prefetch_remainder,
        )

    @classmethod
    def from_transfer_reports(
        cls,
        reports: List[TransferReport],
        flashback_budget_ms: float,
        background_prefetch_remainder: bool = True,
    ) -> "VisionKVPolicy":
        recommendation = recommend_prefetch_block_count(reports, flashback_budget_ms)
        return cls.from_prefetch_budget_recommendation(
            recommendation,
            background_prefetch_remainder=background_prefetch_remainder,
        )
