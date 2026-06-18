"""vLLM-facing hook surface for VisionKV.

This module is intentionally written as an adapter layer rather than a direct
patch against vLLM because the vLLM source tree is not present in this repo.
It captures the shape of the integration we will need in:
- vllm/core/block_manager.py
- vllm/worker/worker.py
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Dict, List, Set

from .controller import VisionKVController


def vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int

    def overlaps(self, other_start: int, other_end: int) -> bool:
        return not (other_end <= self.start or other_start >= self.end)


@dataclass
class SequenceVisionState:
    request_id: str
    vision_token_span: TokenSpan
    vision_block_ids: List[int] = field(default_factory=list)
    offloaded_block_ids: Set[int] = field(default_factory=set)


class VisionBlockMetadataStore:
    """Tracks which logical blocks belong to vision tokens.

    This is the metadata shape we will eventually hang off vLLM's block table.
    """

    def __init__(self) -> None:
        self.sequence_states: Dict[str, SequenceVisionState] = {}
        self.block_to_request: Dict[int, str] = {}

    def register_sequence(self, request_id: str, vision_start: int, vision_end: int) -> None:
        self.sequence_states[request_id] = SequenceVisionState(
            request_id=request_id,
            vision_token_span=TokenSpan(start=vision_start, end=vision_end),
        )

    def record_block_assignment(
        self,
        request_id: str,
        logical_block_id: int,
        token_start: int,
        token_end: int,
    ) -> bool:
        state = self.sequence_states[request_id]
        if not state.vision_token_span.overlaps(token_start, token_end):
            return False

        if logical_block_id not in state.vision_block_ids:
            state.vision_block_ids.append(logical_block_id)
        self.block_to_request[logical_block_id] = request_id
        return True

    def get_vision_block_ids(self, request_id: str) -> List[int]:
        return list(self.sequence_states[request_id].vision_block_ids)

    def get_hot_vision_block_ids(
        self,
        request_id: str,
        hot_block_count: int | None,
    ) -> List[int]:
        vision_block_ids = self.get_vision_block_ids(request_id)
        if hot_block_count is None:
            return vision_block_ids
        return vision_block_ids[:hot_block_count]

    def mark_offloaded(self, request_id: str, logical_block_ids: List[int]) -> None:
        state = self.sequence_states[request_id]
        state.offloaded_block_ids.update(logical_block_ids)

    def mark_prefetched(self, request_id: str, logical_block_ids: List[int]) -> None:
        state = self.sequence_states[request_id]
        state.offloaded_block_ids.difference_update(logical_block_ids)

    def needs_prefetch(self, request_id: str) -> bool:
        return bool(self.sequence_states[request_id].offloaded_block_ids)


class VisionKVVllmAdapter:
    """Adapter that connects decode-time events to the VisionKV controller."""

    def __init__(
        self,
        metadata_store: VisionBlockMetadataStore,
        controller: VisionKVController,
        hot_prefetch_block_count: int | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.controller = controller
        self.hot_prefetch_block_count = hot_prefetch_block_count

    def on_prompt_preprocessed(
        self, request_id: str, vision_token_start: int, vision_token_end: int
    ) -> None:
        """Hook target for multimodal input preprocessing."""

        self.metadata_store.register_sequence(
            request_id=request_id,
            vision_start=vision_token_start,
            vision_end=vision_token_end,
        )

    def on_block_allocated(
        self,
        request_id: str,
        logical_block_id: int,
        token_start: int,
        token_end: int,
    ) -> bool:
        """Hook target for block allocation in vLLM's block manager."""

        return self.metadata_store.record_block_assignment(
            request_id=request_id,
            logical_block_id=logical_block_id,
            token_start=token_start,
            token_end=token_end,
        )

    async def on_decode_step(self, request_id: str, vision_attention_mass: float) -> str:
        """Hook target after each decode step.

        In the real vLLM integration, `vision_attention_mass` will come from the
        Triton/CUDA sidecar that summarizes attention to vision-token blocks.
        """

        await self.controller.observe_decode_step(
            vision_attention_mass,
            trigger_hot_prefetch=False,
        )
        if not self.metadata_store.get_vision_block_ids(request_id):
            return "no-vision-blocks"

        if vision_attention_mass >= self.controller.hot_attention_threshold:
            hot_block_ids = self.metadata_store.get_hot_vision_block_ids(
                request_id,
                self.hot_prefetch_block_count,
            )
            self.controller.request_vision_attention(hot_block_ids)
            return "prefetch-requested"

        vision_blocks_on_gpu = self.controller.block_manager.get_blocks(
            modality="vision",
            location="gpu",
        )
        if not vision_blocks_on_gpu:
            offloaded_ids = self.metadata_store.get_vision_block_ids(request_id)
            self.metadata_store.mark_offloaded(request_id, offloaded_ids)
            return "vision-offloaded"

        return "noop"

    async def before_attention_forward(self, request_id: str) -> bool:
        """Hook target immediately before the attention kernel launches.

        Returns True when we had to stall waiting for prefetched vision blocks.
        """

        if not self.metadata_store.needs_prefetch(request_id):
            return False

        prefetched_ids = self.metadata_store.get_hot_vision_block_ids(
            request_id,
            self.hot_prefetch_block_count,
        )
        stalled = await self.controller.ensure_vision_ready(prefetched_ids)
        self.metadata_store.mark_prefetched(request_id, prefetched_ids)
        return stalled

    def describe_real_integration_points(self) -> Dict[str, str]:
        return {
            "input_preprocessing": (
                "Tag the multimodal token span when vLLM replaces the image "
                "placeholder with projected vision embeddings."
            ),
            "block_manager": (
                "Record logical_block_id -> is_vision_block during block table "
                "allocation and preserve that logical mapping after physical "
                "GPU blocks are released."
            ),
            "decode_loop": (
                "Consume block-level vision attention mass after each decode "
                "step and trigger cold eviction or budgeted hot-set prefetch."
            ),
            "worker_forward": (
                "Before attention runs, wait only if the request still needs "
                "vision blocks and the async prefetch has not completed yet."
            ),
        }
