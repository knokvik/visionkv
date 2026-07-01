#!/usr/bin/env python3
"""
generate_charts.py - Generate professional benchmark charts from results JSON.

Produces PNG images for the README:
  - docs/benchmark_throughput.png
  - docs/benchmark_latency.png
  - docs/benchmark_vram.png
  - docs/benchmark_overview.png  (combined dashboard)

Usage:
    python3 generate_charts.py [--input benchmark_results.json] [--outdir docs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Font / style setup (deferred import so the script can report missing dep)
# ---------------------------------------------------------------------------

mticker = None

def _setup_matplotlib():
    global mticker
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mt
        mticker = mt
    except ImportError:
        print(
            "ERROR: matplotlib is required.  Install with:\n"
            "  pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)

    # Professional style
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#ffffff",
    })
    return plt, mticker


# Colour palette
C_BASELINE = "#2563eb"    # blue
C_VISIONKV = "#dc2626"    # red
C_BASELINE_FILL = "#2563eb"
C_VISIONKV_FILL = "#dc2626"


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def _extract(results: list[dict], key: str) -> list:
    return [r[key] for r in results if r.get("status") == "OK"]


def chart_throughput(plt, baseline: list, visionkv: list, outpath: Path):
    """Line chart: throughput (tok/s) vs concurrent users."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_b = _extract(baseline, "batch_size")
    y_b = _extract(baseline, "throughput_tok_s")
    x_v = _extract(visionkv, "batch_size")
    y_v = _extract(visionkv, "throughput_tok_s")

    ax.plot(x_b, y_b, color=C_BASELINE, marker="o", label="Standard vLLM")
    ax.plot(x_v, y_v, color=C_VISIONKV, marker="s", label="VisionKV")

    ax.set_xlabel("Concurrent Users (Batch Size)")
    ax.set_ylabel("Throughput (tokens / second)")
    ax.set_title("Throughput vs Concurrency")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


def chart_latency(plt, baseline: list, visionkv: list, outpath: Path):
    """Line chart: average latency (s) vs concurrent users."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_b = _extract(baseline, "batch_size")
    y_b = _extract(baseline, "avg_latency_s")
    x_v = _extract(visionkv, "batch_size")
    y_v = _extract(visionkv, "avg_latency_s")

    ax.plot(x_b, y_b, color=C_BASELINE, marker="o", label="Standard vLLM")
    ax.plot(x_v, y_v, color=C_VISIONKV, marker="s", label="VisionKV")

    ax.set_xlabel("Concurrent Users (Batch Size)")
    ax.set_ylabel("Average Latency (seconds)")
    ax.set_title("Latency vs Concurrency")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


def chart_vram(plt, baseline: list, visionkv: list, outpath: Path):
    """Grouped bar chart: peak VRAM at each batch size."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_b = _extract(baseline, "batch_size")
    y_b = _extract(baseline, "vram_after_mib")
    x_v = _extract(visionkv, "batch_size")
    y_v = _extract(visionkv, "vram_after_mib")

    # Use the union of batch sizes; align by index
    import numpy as np
    all_batches = sorted(set(x_b) | set(x_v))
    width = 0.35

    b_map = dict(zip(x_b, y_b))
    v_map = dict(zip(x_v, y_v))

    b_vals = [b_map.get(b, 0) for b in all_batches]
    v_vals = [v_map.get(b, 0) for b in all_batches]

    x_pos = np.arange(len(all_batches))
    ax.bar(x_pos - width / 2, b_vals, width, color=C_BASELINE, alpha=0.85,
           label="Standard vLLM")
    ax.bar(x_pos + width / 2, v_vals, width, color=C_VISIONKV, alpha=0.85,
           label="VisionKV")

    ax.set_xlabel("Concurrent Users (Batch Size)")
    ax.set_ylabel("VRAM (MiB)")
    ax.set_title("Peak VRAM Utilization")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_batches)
    ax.legend(loc="upper left")

    # Format y-axis as thousands
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x / 1000:.0f}K"
    ))

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


def chart_overview(plt, mticker, baseline: list, visionkv: list, outpath: Path):
    """Combined 2x2 dashboard for the top of the README."""
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "VisionKV Concurrency Benchmark - llava-1.5-7b-hf on NVIDIA A6000",
        fontsize=15, fontweight="bold", y=0.98,
    )

    x_b = _extract(baseline, "batch_size")
    x_v = _extract(visionkv, "batch_size")

    # --- (0,0) Throughput ---
    ax = axes[0, 0]
    ax.plot(x_b, _extract(baseline, "throughput_tok_s"),
            color=C_BASELINE, marker="o", label="Standard vLLM")
    ax.plot(x_v, _extract(visionkv, "throughput_tok_s"),
            color=C_VISIONKV, marker="s", label="VisionKV")
    ax.set_xlabel("Concurrent Users")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Throughput")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- (0,1) Latency ---
    ax = axes[0, 1]
    ax.plot(x_b, _extract(baseline, "avg_latency_s"),
            color=C_BASELINE, marker="o", label="Standard vLLM")
    ax.plot(x_v, _extract(visionkv, "avg_latency_s"),
            color=C_VISIONKV, marker="s", label="VisionKV")
    ax.set_xlabel("Concurrent Users")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Average Latency per Request")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- (1,0) VRAM ---
    ax = axes[1, 0]
    b_map = dict(zip(x_b, _extract(baseline, "vram_after_mib")))
    v_map = dict(zip(x_v, _extract(visionkv, "vram_after_mib")))
    all_batches = sorted(set(x_b) | set(x_v))
    b_vals = [b_map.get(b, 0) for b in all_batches]
    v_vals = [v_map.get(b, 0) for b in all_batches]
    x_pos = np.arange(len(all_batches))
    width = 0.35
    ax.bar(x_pos - width / 2, b_vals, width, color=C_BASELINE, alpha=0.85,
           label="Standard vLLM")
    ax.bar(x_pos + width / 2, v_vals, width, color=C_VISIONKV, alpha=0.85,
           label="VisionKV")
    ax.set_xlabel("Concurrent Users")
    ax.set_ylabel("VRAM (MiB)")
    ax.set_title("Peak VRAM Utilization")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_batches)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x / 1000:.0f}K"
    ))

    # --- (1,1) Summary text box ---
    ax = axes[1, 1]
    ax.axis("off")
    b_max = baseline[0].get("max_batch", "N/A") if baseline else "N/A"
    v_max = visionkv[0].get("max_batch", "N/A") if visionkv else "N/A"

    # Find peak throughput
    peak_b = max(_extract(baseline, "throughput_tok_s"), default=0)
    peak_v = max(_extract(visionkv, "throughput_tok_s"), default=0)

    # VRAM at largest batch
    vram_b = _extract(baseline, "vram_after_mib")
    vram_v = _extract(visionkv, "vram_after_mib")
    vram_b_peak = vram_b[-1] if vram_b else 0
    vram_v_peak = vram_v[-1] if vram_v else 0

    summary = (
        f"Benchmark Summary\n"
        f"{'=' * 36}\n\n"
        f"  Model:        llava-hf/llava-1.5-7b-hf\n"
        f"  GPU:          NVIDIA A6000 (48 GB)\n"
        f"  vLLM:         0.23.0\n\n"
        f"  Max Concurrency (Baseline):  {b_max} users\n"
        f"  Max Concurrency (VisionKV):  {v_max} users\n\n"
        f"  Peak Throughput (Baseline):  {peak_b:.1f} tok/s\n"
        f"  Peak Throughput (VisionKV):  {peak_v:.1f} tok/s\n\n"
        f"  Peak VRAM (Baseline):        {vram_b_peak:,.0f} MiB\n"
        f"  Peak VRAM (VisionKV):        {vram_v_peak:,.0f} MiB\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0",
                      edgecolor="#cccccc"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts from results JSON."
    )
    parser.add_argument("--input", default="benchmark_results.json",
                        help="Path to benchmark_results.json")
    parser.add_argument("--outdir", default="docs",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        print(f"ERROR: {inpath} not found", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    with open(inpath) as f:
        data = json.load(f)

    baseline = data["baseline"]["results"]
    visionkv = data["visionkv"]["results"]

    plt, mticker = _setup_matplotlib()

    print(f"Generating charts from {inpath} -> {outdir}/")

    chart_throughput(plt, baseline, visionkv, outdir / "benchmark_throughput.png")
    chart_latency(plt, baseline, visionkv, outdir / "benchmark_latency.png")
    chart_vram(plt, baseline, visionkv, outdir / "benchmark_vram.png")
    chart_overview(plt, mticker, baseline, visionkv, outdir / "benchmark_overview.png")

    print("Done.")


if __name__ == "__main__":
    main()
