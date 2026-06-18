"""Tests for the PyTorch prototype helpers."""

from __future__ import annotations

import unittest

from visionkv.pytorch_prototype import (
    FLOAT32_BYTES,
    PrototypeConfig,
    TorchVisionKVPrototype,
    format_bytes,
    megabytes_to_numel,
    resolve_prefetch_block_count,
    should_use_non_blocking_copy,
    torch_available,
)


class HelperTests(unittest.TestCase):
    def test_megabytes_to_numel_uses_float32_by_default(self) -> None:
        expected = (64 * 1024 * 1024) // FLOAT32_BYTES
        self.assertEqual(megabytes_to_numel(64), expected)

    def test_format_bytes(self) -> None:
        self.assertEqual(format_bytes(2 * 1024 * 1024), "2.0MB")

    def test_non_blocking_copy_is_used_for_cpu_cuda_transfers_only(self) -> None:
        self.assertTrue(should_use_non_blocking_copy("cuda", "cpu", True))
        self.assertTrue(should_use_non_blocking_copy("cpu", "cuda", True))
        self.assertFalse(should_use_non_blocking_copy("cuda", "cuda", True))
        self.assertFalse(should_use_non_blocking_copy("cpu", "cpu", True))
        self.assertFalse(should_use_non_blocking_copy("cuda", "cpu", False))

    def test_resolve_prefetch_block_count_defaults_to_all_blocks(self) -> None:
        self.assertEqual(resolve_prefetch_block_count(10, None), 10)

    def test_resolve_prefetch_block_count_clamps_to_available_blocks(self) -> None:
        self.assertEqual(resolve_prefetch_block_count(10, 2), 2)
        self.assertEqual(resolve_prefetch_block_count(10, 20), 10)

    def test_resolve_prefetch_block_count_rejects_non_positive_requests(self) -> None:
        with self.assertRaises(ValueError):
            resolve_prefetch_block_count(10, 0)


@unittest.skipUnless(torch_available(), "PyTorch is not installed")
class PrototypeRuntimeTests(unittest.TestCase):
    def test_select_runtime_devices_supports_auto(self) -> None:
        devices = TorchVisionKVPrototype._select_runtime_devices("auto")
        self.assertIn(devices.accelerator, {"cpu", "cuda", "mps"})

    def test_prototype_can_allocate_small_block(self) -> None:
        prototype = TorchVisionKVPrototype(
            PrototypeConfig(
                num_vision_blocks=1,
                vision_block_mb=1,
                num_text_blocks=1,
                text_block_mb=1,
                overlap_matmul_dim=16,
                preferred_device="cpu",
            )
        )
        block = prototype.allocate_block("vision", 1)
        self.assertEqual(block.modality, "vision")
        self.assertEqual(block.location, "cpu")

    def test_cpu_runtime_does_not_create_pinned_staging_pool(self) -> None:
        prototype = TorchVisionKVPrototype(
            PrototypeConfig(
                num_vision_blocks=1,
                vision_block_mb=1,
                num_text_blocks=1,
                text_block_mb=1,
                overlap_matmul_dim=16,
                preferred_device="cpu",
            )
        )
        block = prototype.allocate_block("vision", 1)
        self.assertIsNone(block.cpu_staging_tensor)


if __name__ == "__main__":
    unittest.main()
