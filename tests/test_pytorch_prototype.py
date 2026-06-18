"""Tests for the PyTorch prototype helpers."""

from __future__ import annotations

import unittest

from visionkv.pytorch_prototype import (
    FLOAT32_BYTES,
    PrototypeConfig,
    TorchVisionKVPrototype,
    format_bytes,
    megabytes_to_numel,
    torch_available,
)


class HelperTests(unittest.TestCase):
    def test_megabytes_to_numel_uses_float32_by_default(self) -> None:
        expected = (64 * 1024 * 1024) // FLOAT32_BYTES
        self.assertEqual(megabytes_to_numel(64), expected)

    def test_format_bytes(self) -> None:
        self.assertEqual(format_bytes(2 * 1024 * 1024), "2.0MB")


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


if __name__ == "__main__":
    unittest.main()
