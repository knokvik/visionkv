"""Tests for the live vLLM integration plugin logic."""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from dataclasses import dataclass


def _install_fake_vllm_modules() -> None:
    torch_module = types.ModuleType("torch")
    vllm_module = types.ModuleType("vllm")
    engine_module = types.ModuleType("vllm.engine")
    engine_llm_engine_module = types.ModuleType("vllm.engine.llm_engine")
    multimodal_module = types.ModuleType("vllm.multimodal")
    multimodal_inputs_module = types.ModuleType("vllm.multimodal.inputs")
    v1_module = types.ModuleType("vllm.v1")
    v1_engine_module = types.ModuleType("vllm.v1.engine")
    v1_worker_module = types.ModuleType("vllm.v1.worker")
    v1_worker_base_module = types.ModuleType("vllm.v1.worker.worker_base")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def memory_reserved() -> int:
            return 0

        @staticmethod
        def max_memory_reserved() -> int:
            return 0

    torch_module.cuda = FakeCuda()
    class FakeLLMEngine:
        pass

    @dataclass
    class FakePlaceholderRange:
        offset: int
        length: int

        def extract_embeds_range(self) -> list[tuple[int, int]]:
            return [(self.offset, self.offset + self.length - 1)]

    @dataclass
    class FakeFeature:
        modality: str
        mm_position: FakePlaceholderRange

    @dataclass
    class FakeEngineCoreRequest:
        request_id: str
        prompt_token_ids: list[int]
        mm_features: list[FakeFeature] | None = None

    class FakeWorkerBase:
        pass

    engine_llm_engine_module.LLMEngine = FakeLLMEngine
    multimodal_inputs_module.PlaceholderRange = FakePlaceholderRange
    multimodal_inputs_module.MultiModalFeatureSpec = FakeFeature
    v1_engine_module.EngineCoreRequest = FakeEngineCoreRequest
    v1_worker_base_module.WorkerBase = FakeWorkerBase

    sys.modules["torch"] = torch_module
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.engine"] = engine_module
    sys.modules["vllm.engine.llm_engine"] = engine_llm_engine_module
    sys.modules["vllm.multimodal"] = multimodal_module
    sys.modules["vllm.multimodal.inputs"] = multimodal_inputs_module
    sys.modules["vllm.v1"] = v1_module
    sys.modules["vllm.v1.engine"] = v1_engine_module
    sys.modules["vllm.v1.worker"] = v1_worker_module
    sys.modules["vllm.v1.worker.worker_base"] = v1_worker_base_module


def _load_live_integration():
    _install_fake_vllm_modules()
    sys.modules.pop("visionkv.live_integration", None)
    return importlib.import_module("visionkv.live_integration")


class LiveIntegrationTests(unittest.TestCase):
    def test_step_lifecycle_offloads_when_generation_crosses_threshold(self) -> None:
        live = _load_live_integration()

        class FakeInputProcessor:
            def process_inputs(self, *args, **kwargs):
                raise AssertionError("process_inputs should not be called in this test")

        class FakeEngine(live.LLMEngine):
            def __init__(self, step_outputs):
                self.input_processor = FakeInputProcessor()
                self._step_outputs = step_outputs

            def add_request(self, *args, **kwargs):
                raise AssertionError("add_request should not be called in this test")

            def step(self):
                return self._step_outputs

        request = live.EngineCoreRequest(
            request_id="req-1",
            prompt_token_ids=list(range(64)),
            mm_features=[
                sys.modules["vllm.multimodal.inputs"].MultiModalFeatureSpec(
                    modality="image",
                    mm_position=live.PlaceholderRange(offset=0, length=64),
                )
            ],
        )

        class FakeSequenceOutput:
            def __init__(self, token_ids):
                self.token_ids = token_ids

        class FakeRequestOutput:
            def __init__(self, request_id, token_ids):
                self.request_id = request_id
                self.outputs = [FakeSequenceOutput(token_ids)]

        engine = FakeEngine([FakeRequestOutput("req-1-internal", list(range(60)))])
        plugin = live.VisionKVPlugin(engine, offload_after_generation_tokens=50).install()
        plugin.tag_vision_blocks_hook(request, {"prompt": "<image> hello"})
        plugin.metadata_store.finalize_request_id("req-1", "req-1-internal")

        engine.step()

        state = plugin.metadata_store.get("req-1-internal")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.generated_tokens, 60)
        self.assertEqual(sorted(state.offloaded_block_ids), [0, 1, 2, 3])
        self.assertEqual(state.offload_count, 1)
        self.assertFalse(state.pending_offload)

    def test_followup_prefetch_restores_hot_set_then_background_remainder(self) -> None:
        live = _load_live_integration()
        plugin = live.VisionKVPlugin(object())

        request = live.EngineCoreRequest(
            request_id="req-2",
            prompt_token_ids=list(range(96)),
            mm_features=[
                sys.modules["vllm.multimodal.inputs"].MultiModalFeatureSpec(
                    modality="image",
                    mm_position=live.PlaceholderRange(offset=0, length=96),
                )
            ],
        )
        plugin.tag_vision_blocks_hook(request, {"prompt": "<image> inspect"})
        plugin.metadata_store.finalize_request_id("req-2", "req-2-internal")
        state = plugin.metadata_store.get("req-2-internal")
        assert state is not None
        state.generated_tokens = 100
        plugin.controller.maybe_mark_for_offload(state)
        plugin._advance_lifecycle()

        plugin._trigger_followup_prefetch("new-request")

        self.assertEqual(sorted(state.hot_prefetched_block_ids), [0, 1])
        self.assertEqual(sorted(state.offloaded_block_ids), [2, 3, 4, 5])
        self.assertTrue(state.background_prefetch_pending)

        plugin._advance_lifecycle()

        self.assertEqual(sorted(state.background_prefetched_block_ids), [2, 3, 4, 5])
        self.assertFalse(state.offloaded_block_ids)
        self.assertFalse(state.background_prefetch_pending)
        self.assertEqual(plugin.snapshot()["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
