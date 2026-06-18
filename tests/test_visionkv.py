"""Tests for the VisionKV mock."""

from __future__ import annotations

import asyncio
import unittest

from visionkv.block_manager import MockBlockSpaceManager
from visionkv.controller import VisionKVController


class BlockManagerTests(unittest.TestCase):
    def test_offload_preserves_logical_entry_and_frees_gpu_slot(self) -> None:
        manager = MockBlockSpaceManager()
        vision_block = manager.allocate_block(modality="vision", tensor_size_mb=256)

        freed_mb = manager.offload_blocks([vision_block.logical_id])
        stored_block = manager.block_table[vision_block.logical_id]

        self.assertEqual(freed_mb, 256)
        self.assertEqual(stored_block.location, "cpu")
        self.assertIsNone(stored_block.physical_block_id)
        self.assertEqual(stored_block.cpu_slot_id, f"cpu:{vision_block.logical_id}")
        self.assertEqual(manager.free_gpu_block_count(), 1)

    def test_prefetch_rebinds_logical_block_to_gpu(self) -> None:
        manager = MockBlockSpaceManager()
        vision_block = manager.allocate_block(modality="vision", tensor_size_mb=256)
        original_physical_id = vision_block.physical_block_id

        manager.offload_blocks([vision_block.logical_id])
        restored_mb = manager.prefetch_blocks([vision_block.logical_id])
        stored_block = manager.block_table[vision_block.logical_id]

        self.assertEqual(restored_mb, 256)
        self.assertEqual(stored_block.location, "gpu")
        self.assertIsNotNone(stored_block.physical_block_id)
        self.assertEqual(stored_block.physical_block_id, original_physical_id)
        self.assertIsNone(stored_block.cpu_slot_id)


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_eviction_requires_cold_streak_and_text_pressure(self) -> None:
        manager = MockBlockSpaceManager()
        for _ in range(10):
            manager.allocate_block(modality="vision", tensor_size_mb=256)
        for _ in range(21):
            manager.allocate_block(modality="text", tensor_size_mb=64)

        controller = VisionKVController(
            manager,
            text_eviction_threshold=20,
            cold_steps_required=3,
            offload_delay_s=0.0,
            prefetch_delay_s=0.0,
        )

        await controller.observe_decode_step(0.04)
        await controller.observe_decode_step(0.03)
        self.assertEqual(len(manager.get_blocks(modality="vision", location="gpu")), 10)

        await controller.observe_decode_step(0.01)
        self.assertEqual(len(manager.get_blocks(modality="vision", location="gpu")), 0)
        self.assertEqual(len(manager.get_blocks(modality="vision", location="cpu")), 10)

    async def test_prefetch_can_be_hidden_before_kernel_wait(self) -> None:
        manager = MockBlockSpaceManager()
        for _ in range(3):
            manager.allocate_block(modality="vision", tensor_size_mb=128)
        manager.offload_blocks([block.logical_id for block in manager.get_blocks(modality="vision")])

        controller = VisionKVController(
            manager,
            offload_delay_s=0.0,
            prefetch_delay_s=0.0,
        )

        await controller.observe_decode_step(0.25)
        stalled = await controller.ensure_vision_ready()

        self.assertFalse(stalled)
        self.assertEqual(len(manager.get_blocks(modality="vision", location="gpu")), 3)

    async def test_wait_event_stalls_if_prefetch_is_not_ready(self) -> None:
        manager = MockBlockSpaceManager()
        for _ in range(2):
            manager.allocate_block(modality="vision", tensor_size_mb=128)
        manager.offload_blocks([block.logical_id for block in manager.get_blocks(modality="vision")])

        controller = VisionKVController(
            manager,
            offload_delay_s=0.0,
            prefetch_delay_s=0.02,
        )

        await controller.observe_decode_step(0.25)
        stalled = await controller.ensure_vision_ready()

        self.assertTrue(stalled)
        self.assertEqual(len(manager.get_blocks(modality="vision", location="gpu")), 2)


if __name__ == "__main__":
    unittest.main()
