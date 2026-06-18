"""Mock VisionKV controller with cold/hot attention heuristics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from .block_manager import MockBlockSpaceManager


@dataclass
class DecodeEvent:
    step: int
    vision_attention: float
    action: str


class VisionKVController:
    """Coordinates cold eviction and hot prefetch of vision blocks."""

    def __init__(
        self,
        block_manager: MockBlockSpaceManager,
        text_eviction_threshold: int = 20,
        cold_attention_threshold: float = 0.05,
        hot_attention_threshold: float = 0.20,
        cold_steps_required: int = 3,
        offload_delay_s: float = 0.05,
        prefetch_delay_s: float = 0.2,
    ) -> None:
        self.block_manager = block_manager
        self.text_eviction_threshold = text_eviction_threshold
        self.cold_attention_threshold = cold_attention_threshold
        self.hot_attention_threshold = hot_attention_threshold
        self.cold_steps_required = cold_steps_required
        self.offload_delay_s = offload_delay_s
        self.prefetch_delay_s = prefetch_delay_s

        self.decode_step = 0
        self.cold_streak = 0
        self.events: List[DecodeEvent] = []
        self._prefetch_task: Optional[asyncio.Task[None]] = None
        self._prefetch_target_ids: Optional[List[int]] = None
        self._background_prefetch_task: Optional[asyncio.Task[None]] = None
        self._background_prefetch_target_ids: Optional[List[int]] = None

    async def observe_decode_step(
        self,
        vision_attention: float,
        trigger_hot_prefetch: bool = True,
    ) -> None:
        self.decode_step += 1
        action = "noop"

        if vision_attention < self.cold_attention_threshold:
            self.cold_streak += 1
        else:
            self.cold_streak = 0

        if self._should_evict():
            await self._offload_vision_blocks()
            action = "evict"
        elif vision_attention >= self.hot_attention_threshold and trigger_hot_prefetch:
            self.request_vision_attention()
            action = "prefetch-requested"

        self.events.append(
            DecodeEvent(
                step=self.decode_step,
                vision_attention=vision_attention,
                action=action,
            )
        )

    def request_vision_attention(
        self,
        logical_block_ids: Optional[List[int]] = None,
        continuation_logical_block_ids: Optional[List[int]] = None,
    ) -> None:
        target_ids = self._resolve_prefetch_target_ids(logical_block_ids)
        continuation_ids = self._resolve_prefetch_target_ids(continuation_logical_block_ids)
        if not target_ids and not continuation_ids:
            return

        if target_ids and (self._prefetch_task is None or self._prefetch_task.done()):
            self._prefetch_target_ids = target_ids
            self._prefetch_task = asyncio.create_task(self._prefetch_vision_blocks(target_ids))

        if continuation_ids and (
            self._background_prefetch_task is None or self._background_prefetch_task.done()
        ):
            self._background_prefetch_target_ids = continuation_ids
            self._background_prefetch_task = asyncio.create_task(
                self._prefetch_after_current(continuation_ids)
            )

    async def ensure_vision_ready(self, logical_block_ids: Optional[List[int]] = None) -> bool:
        """Wait for in-flight prefetch if the next attention step needs vision blocks.

        Returns True when the caller had to stall for the prefetch to complete.
        """

        if not self._resolve_prefetch_target_ids(logical_block_ids):
            return False

        if self._prefetch_task is None:
            return False

        if self._prefetch_task.done():
            await self._prefetch_task
            return False

        # Give a just-scheduled prefetch one loop turn to finish before
        # counting the wait as a visible stall.
        await asyncio.sleep(0)
        if self._prefetch_task.done():
            await self._prefetch_task
            return False

        await self._prefetch_task
        return True

    async def ensure_background_prefetch_complete(
        self,
        logical_block_ids: Optional[List[int]] = None,
    ) -> bool:
        """Wait for any queued background remainder prefetch."""

        if not self._resolve_prefetch_target_ids(logical_block_ids):
            return False

        if self._background_prefetch_task is None:
            return False

        if self._background_prefetch_task.done():
            await self._background_prefetch_task
            return False

        await self._background_prefetch_task
        return True

    def _resolve_prefetch_target_ids(self, logical_block_ids: Optional[List[int]]) -> List[int]:
        if logical_block_ids is None:
            return [
                block.logical_id
                for block in self.block_manager.get_blocks(modality="vision", location="cpu")
            ]

        return [
            logical_id
            for logical_id in logical_block_ids
            if self.block_manager.block_table[logical_id].location == "cpu"
        ]

    def _should_evict(self) -> bool:
        return (
            self.block_manager.text_block_count() > self.text_eviction_threshold
            and self.cold_streak >= self.cold_steps_required
            and bool(self.block_manager.get_blocks(modality="vision", location="gpu"))
        )

    async def _offload_vision_blocks(self) -> None:
        if self.offload_delay_s > 0:
            await asyncio.sleep(self.offload_delay_s)
        logical_ids = [
            block.logical_id
            for block in self.block_manager.get_blocks(modality="vision", location="gpu")
        ]
        if not logical_ids:
            return
        freed_mb = self.block_manager.offload_blocks(logical_ids)
        print(
            f"[evict] step={self.decode_step} cold_streak={self.cold_streak} "
            f"freed={freed_mb}MB | {self.block_manager.summary()}"
        )

    async def _prefetch_vision_blocks(self, logical_ids: Optional[List[int]] = None) -> None:
        if self.prefetch_delay_s > 0:
            await asyncio.sleep(self.prefetch_delay_s)
        logical_ids = self._resolve_prefetch_target_ids(logical_ids)
        if not logical_ids:
            return
        restored_mb = self.block_manager.prefetch_blocks(logical_ids)
        print(
            f"[prefetch] step={self.decode_step} restored={restored_mb}MB | "
            f"{self.block_manager.summary()}"
        )

    async def _prefetch_after_current(self, logical_ids: List[int]) -> None:
        if self._prefetch_task is not None:
            await self._prefetch_task
        # Let the caller proceed with the hot-set first so the background
        # remainder behaves like a lower-priority continuation.
        await asyncio.sleep(0)
        await self._prefetch_vision_blocks(logical_ids)
