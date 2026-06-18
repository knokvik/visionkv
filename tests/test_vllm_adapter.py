"""Tests for the vLLM integration adapter."""

from __future__ import annotations

import unittest

from visionkv.block_manager import MockBlockSpaceManager
from visionkv.controller import VisionKVController
from visionkv.policy import VisionKVPolicy
from visionkv.vllm_adapter import VisionBlockMetadataStore, VisionKVVllmAdapter


class MetadataStoreTests(unittest.TestCase):
    def test_record_block_assignment_marks_only_overlapping_blocks(self) -> None:
        store = VisionBlockMetadataStore()
        store.register_sequence("req-1", vision_start=32, vision_end=96)

        vision_hit = store.record_block_assignment("req-1", 7, token_start=32, token_end=48)
        text_hit = store.record_block_assignment("req-1", 8, token_start=96, token_end=112)

        self.assertTrue(vision_hit)
        self.assertFalse(text_hit)
        self.assertEqual(store.get_vision_block_ids("req-1"), [7])

    def test_hot_vision_block_ids_preserve_allocation_order(self) -> None:
        store = VisionBlockMetadataStore()
        store.register_sequence("req-1", vision_start=0, vision_end=64)
        store.record_block_assignment("req-1", 4, token_start=16, token_end=32)
        store.record_block_assignment("req-1", 1, token_start=0, token_end=16)
        store.record_block_assignment("req-1", 9, token_start=32, token_end=48)

        self.assertEqual(store.get_vision_block_ids("req-1"), [4, 1, 9])
        self.assertEqual(store.get_hot_vision_block_ids("req-1", 2), [4, 1])
        self.assertEqual(store.get_cold_vision_block_ids("req-1", 2), [9])


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_tracks_offload_and_prefetch_state(self) -> None:
        manager = MockBlockSpaceManager()
        for _ in range(3):
            manager.allocate_block(modality="vision", tensor_size_mb=128)
        for _ in range(21):
            manager.allocate_block(modality="text", tensor_size_mb=64)

        controller = VisionKVController(
            manager,
            text_eviction_threshold=20,
            cold_steps_required=2,
            offload_delay_s=0.0,
            prefetch_delay_s=0.0,
        )
        store = VisionBlockMetadataStore()
        adapter = VisionKVVllmAdapter(
            store,
            controller,
            policy=VisionKVPolicy(background_prefetch_remainder=False),
        )

        adapter.on_prompt_preprocessed("req-1", vision_token_start=0, vision_token_end=48)
        adapter.on_block_allocated("req-1", logical_block_id=0, token_start=0, token_end=16)
        adapter.on_block_allocated("req-1", logical_block_id=1, token_start=16, token_end=32)
        adapter.on_block_allocated("req-1", logical_block_id=2, token_start=32, token_end=48)

        status = await adapter.on_decode_step("req-1", 0.01)
        self.assertEqual(status, "noop")

        status = await adapter.on_decode_step("req-1", 0.01)
        self.assertEqual(status, "vision-offloaded")
        self.assertTrue(store.needs_prefetch("req-1"))

        status = await adapter.on_decode_step("req-1", 0.30)
        self.assertEqual(status, "prefetch-requested")

        stalled = await adapter.before_attention_forward("req-1")
        self.assertFalse(stalled)
        self.assertFalse(store.needs_prefetch("req-1"))

    async def test_adapter_prefetches_only_budgeted_hot_subset(self) -> None:
        manager = MockBlockSpaceManager()
        for _ in range(4):
            manager.allocate_block(modality="vision", tensor_size_mb=128)
        for _ in range(21):
            manager.allocate_block(modality="text", tensor_size_mb=64)

        controller = VisionKVController(
            manager,
            text_eviction_threshold=20,
            cold_steps_required=2,
            offload_delay_s=0.0,
            prefetch_delay_s=0.0,
        )
        store = VisionBlockMetadataStore()
        adapter = VisionKVVllmAdapter(
            store,
            controller,
            policy=VisionKVPolicy(
                hot_prefetch_block_count=2,
                background_prefetch_remainder=True,
            ),
        )

        adapter.on_prompt_preprocessed("req-hot", vision_token_start=0, vision_token_end=64)
        adapter.on_block_allocated("req-hot", logical_block_id=0, token_start=0, token_end=16)
        adapter.on_block_allocated("req-hot", logical_block_id=1, token_start=16, token_end=32)
        adapter.on_block_allocated("req-hot", logical_block_id=2, token_start=32, token_end=48)
        adapter.on_block_allocated("req-hot", logical_block_id=3, token_start=48, token_end=64)

        await adapter.on_decode_step("req-hot", 0.01)
        status = await adapter.on_decode_step("req-hot", 0.01)
        self.assertEqual(status, "vision-offloaded")
        self.assertEqual(store.get_vision_block_ids("req-hot"), [0, 1, 2, 3])

        status = await adapter.on_decode_step("req-hot", 0.30)
        self.assertEqual(status, "prefetch-requested")

        stalled = await adapter.before_attention_forward("req-hot")
        self.assertFalse(stalled)
        self.assertEqual(
            len(manager.get_blocks(modality="vision", location="gpu")),
            2,
        )
        self.assertEqual(
            sorted(store.sequence_states["req-hot"].offloaded_block_ids),
            [2, 3],
        )

        stalled = await adapter.complete_background_prefetch("req-hot")
        self.assertTrue(stalled)
        self.assertEqual(
            len(manager.get_blocks(modality="vision", location="gpu")),
            4,
        )
        self.assertFalse(store.needs_prefetch("req-hot"))

    async def test_before_attention_forward_skips_when_request_has_no_offloaded_blocks(self) -> None:
        manager = MockBlockSpaceManager()
        controller = VisionKVController(manager, offload_delay_s=0.0, prefetch_delay_s=0.0)
        store = VisionBlockMetadataStore()
        adapter = VisionKVVllmAdapter(store, controller)

        adapter.on_prompt_preprocessed("req-2", vision_token_start=10, vision_token_end=20)
        stalled = await adapter.before_attention_forward("req-2")
        self.assertFalse(stalled)


if __name__ == "__main__":
    unittest.main()
