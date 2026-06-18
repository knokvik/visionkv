"""Tests for benchmark comparison helpers."""

from __future__ import annotations

import unittest

from visionkv.benchmark_harness import BenchmarkSample, compare_samples


class BenchmarkHarnessTests(unittest.TestCase):
    def test_compare_samples_computes_expected_deltas(self) -> None:
        baseline = BenchmarkSample(
            label="baseline",
            max_batch_size=4,
            p50_latency_ms=120.0,
            p95_latency_ms=180.0,
            flashback_latency_ms=90.0,
            peak_gpu_memory_mb=32000,
        )
        candidate = BenchmarkSample(
            label="visionkv",
            max_batch_size=12,
            p50_latency_ms=100.0,
            p95_latency_ms=150.0,
            flashback_latency_ms=40.0,
            peak_gpu_memory_mb=18000,
        )

        comparison = compare_samples(baseline, candidate)
        self.assertEqual(comparison.batch_size_gain_x, 3.0)
        self.assertEqual(comparison.p50_latency_delta_ms, -20.0)
        self.assertEqual(comparison.p95_latency_delta_ms, -30.0)
        self.assertEqual(comparison.flashback_latency_delta_ms, -50.0)
        self.assertEqual(comparison.peak_gpu_memory_saved_mb, 14000)


if __name__ == "__main__":
    unittest.main()
