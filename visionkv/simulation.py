"""Runnable simulation for the richer VisionKV mock."""

from __future__ import annotations

import asyncio

from .block_manager import MockBlockSpaceManager
from .controller import VisionKVController


async def run_simulation() -> None:
    manager = MockBlockSpaceManager()
    controller = VisionKVController(manager)

    print("== VisionKV Simulation v2 ==")
    print("Phase 1: image prefill allocates 10 logical vision blocks")
    for _ in range(10):
        block = manager.allocate_block(modality="vision", tensor_size_mb=256)
        print(
            f"allocated logical={block.logical_id:02d} modality=vision "
            f"physical={block.physical_block_id:02d}"
        )

    print("\nPhase 2: decode allocates text blocks while vision attention cools")
    for decode_index in range(1, 26):
        block = manager.allocate_block(modality="text", tensor_size_mb=64)
        attention = 0.12 if decode_index < 18 else 0.02
        print(
            f"decode={decode_index:02d} text_logical={block.logical_id:02d} "
            f"vision_attention={attention:.2f}"
        )
        await controller.observe_decode_step(attention)

    print(f"\nAfter cooling: {manager.summary()}")
    sample_mapping = [
        (
            block.logical_id,
            block.physical_block_id,
            block.cpu_slot_id,
            block.location,
        )
        for block in manager.get_blocks(modality="vision")[:3]
    ]
    print(f"Sample logical mappings after eviction: {sample_mapping}")

    print("\nPhase 3: follow-up question spikes attention before the next kernel")
    await controller.observe_decode_step(0.31)
    await asyncio.sleep(0.25)
    stalled = await controller.ensure_vision_ready()
    print(f"wait_event_stalled={stalled}")
    print(f"Final state: {manager.summary()}")


def main() -> None:
    asyncio.run(run_simulation())


if __name__ == "__main__":
    main()
