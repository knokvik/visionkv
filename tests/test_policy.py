"""Tests for shared VisionKV policy configuration."""

from __future__ import annotations

import unittest

from visionkv.policy import VisionKVPolicy
from visionkv.pytorch_prototype import TransferReport


class VisionKVPolicyTests(unittest.TestCase):
    def test_policy_can_be_derived_from_transfer_reports(self) -> None:
        reports = [
            TransferReport("cpu->cuda", 256 * 1024 * 1024, 0.5, 22.0, 0.3, True, 1, False),
            TransferReport("cpu->cuda", 512 * 1024 * 1024, 0.8, 44.0, 0.3, True, 2, False),
            TransferReport("cpu->cuda", 1024 * 1024 * 1024, 1.1, 87.0, 0.3, True, 4, False),
        ]

        policy = VisionKVPolicy.from_transfer_reports(
            reports,
            flashback_budget_ms=50.0,
            background_prefetch_remainder=True,
        )

        self.assertEqual(policy.hot_prefetch_block_count, 2)
        self.assertEqual(policy.flashback_budget_ms, 50.0)
        self.assertTrue(policy.background_prefetch_remainder)
