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

    async def observe_decode_step(self, vision_attention: float) -> None:
        self.decode_step += 1
        action = "noop"

        if vision_attention < self.cold_attention_threshold:
            self.cold_streak += 1
        else:
            self.cold_streak = 0

        if self._should_evict():
            await self._offload_vision_blocks()
            action = "evict"
        elif vision_attention >= self.hot_attention_threshold:
            self.request_vision_attention()
            action = "prefetch-requested"

        self.events.append(
            DecodeEvent(
                step=self.decode_step,
                vision_attention=vision_attention,
                action=action,
            )
        )

    def request_vision_attention(self) -> None:
        vision_cpu_blocks = self.block_manager.get_blocks(modality="vision", location="cpu")
        if not vision_cpu_blocks:
            return

        if self._prefetch_task is None or self._prefetch_task.done():
            self._prefetch_task = asyncio.create_task(self._prefetch_vision_blocks())

    async def ensure_vision_ready(self) -> bool:
        """Wait for in-flight prefetch if the next attention step needs vision blocks.

        Returns True when the caller had to stall for the prefetch to complete.
        """

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

    async def _prefetch_vision_blocks(self) -> None:
        if self.prefetch_delay_s > 0:
            await asyncio.sleep(self.prefetch_delay_s)
        logical_ids = [
            block.logical_id
            for block in self.block_manager.get_blocks(modality="vision", location="cpu")
        ]
        if not logical_ids:
            return
        restored_mb = self.block_manager.prefetch_blocks(logical_ids)
        print(
            f"[prefetch] step={self.decode_step} restored={restored_mb}MB | "
            f"{self.block_manager.summary()}"
        )
