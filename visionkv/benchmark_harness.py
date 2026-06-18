"""Benchmark helpers for baseline-vs-VisionKV comparisons."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Optional


@dataclass
class BenchmarkSample:
    label: str
    max_batch_size: int
    p50_latency_ms: float
    p95_latency_ms: float
    flashback_latency_ms: float
    peak_gpu_memory_mb: int


@dataclass
class BenchmarkComparison:
    baseline_label: str
    candidate_label: str
    batch_size_gain_x: float
    p50_latency_delta_ms: float
    p95_latency_delta_ms: float
    flashback_latency_delta_ms: float
    peak_gpu_memory_saved_mb: int


def compare_samples(
    baseline: BenchmarkSample, candidate: BenchmarkSample
) -> BenchmarkComparison:
    return BenchmarkComparison(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        batch_size_gain_x=(
            candidate.max_batch_size / baseline.max_batch_size
            if baseline.max_batch_size
            else 0.0
        ),
        p50_latency_delta_ms=candidate.p50_latency_ms - baseline.p50_latency_ms,
        p95_latency_delta_ms=candidate.p95_latency_ms - baseline.p95_latency_ms,
        flashback_latency_delta_ms=(
            candidate.flashback_latency_ms - baseline.flashback_latency_ms
        ),
        peak_gpu_memory_saved_mb=baseline.peak_gpu_memory_mb - candidate.peak_gpu_memory_mb,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare benchmark outputs.")
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-max-batch-size", type=int, required=True)
    parser.add_argument("--baseline-p50-ms", type=float, required=True)
    parser.add_argument("--baseline-p95-ms", type=float, required=True)
    parser.add_argument("--baseline-flashback-ms", type=float, required=True)
    parser.add_argument("--baseline-peak-gpu-mb", type=int, required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--candidate-max-batch-size", type=int, required=True)
    parser.add_argument("--candidate-p50-ms", type=float, required=True)
    parser.add_argument("--candidate-p95-ms", type=float, required=True)
    parser.add_argument("--candidate-flashback-ms", type=float, required=True)
    parser.add_argument("--candidate-peak-gpu-mb", type=int, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline = BenchmarkSample(
        label=args.baseline_label,
        max_batch_size=args.baseline_max_batch_size,
        p50_latency_ms=args.baseline_p50_ms,
        p95_latency_ms=args.baseline_p95_ms,
        flashback_latency_ms=args.baseline_flashback_ms,
        peak_gpu_memory_mb=args.baseline_peak_gpu_mb,
    )
    candidate = BenchmarkSample(
        label=args.candidate_label,
        max_batch_size=args.candidate_max_batch_size,
        p50_latency_ms=args.candidate_p50_ms,
        p95_latency_ms=args.candidate_p95_ms,
        flashback_latency_ms=args.candidate_flashback_ms,
        peak_gpu_memory_mb=args.candidate_peak_gpu_mb,
    )
    comparison = compare_samples(baseline, candidate)
    print(json.dumps(asdict(comparison), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
