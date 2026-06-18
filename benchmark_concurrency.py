#!/usr/bin/env python3
"""
benchmark_concurrency.py — VisionKV Concurrency Benchmark
==========================================================

Proves VisionKV's memory savings by comparing maximum concurrent VLM
users with and without the VisionKV eviction plugin active.

Protocol:
  1. Start a vLLM engine with llava-hf/llava-1.5-7b-hf.
  2. Create N prompts, each with a dummy 448×448 image, requesting 100 tokens.
  3. Submit them as a single batch via llm.generate() — vLLM's scheduler
     handles true concurrency inside the engine.
  4. Increase batch size (5 → 10 → 15 → 20 → …) until OOM.
  5. Run once WITHOUT VisionKV (baseline) and once WITH VisionKV.
  6. Print the comparison.

Usage on remote GPU:
  python3 benchmark_concurrency.py
  python3 benchmark_concurrency.py --gpu-mem-util 0.40 --max-batch 40

The --gpu-mem-util flag constrains the KV-cache pool so OOM occurs at
realistic batch sizes. Lower values = tighter memory = earlier OOM.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import textwrap
import time
import traceback

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]

PROMPT_QUESTIONS = [
    "Describe every detail you can see in this image.",
    "What colors and patterns are visible in this image?",
    "What geometric shapes can you identify in this image?",
    "What is the dominant visual element in this image?",
    "Describe the texture and composition of this image.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_dummy_image(size: int = 448):
    """Create a deterministic 448×448 purple-and-white checkerboard PNG."""
    from PIL import Image
    import numpy as np

    arr = np.zeros((size, size, 3), dtype=np.uint8)
    block = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                arr[i * block : (i + 1) * block,
                    j * block : (j + 1) * block] = [128, 0, 128]
            else:
                arr[i * block : (i + 1) * block,
                    j * block : (j + 1) * block] = [255, 255, 255]
    return Image.fromarray(arr)


def build_prompts(n: int, image):
    """Build n multimodal prompt dicts for vLLM offline generation."""
    prompts = []
    # Generate a prompt of roughly 2500 words (~2600 tokens), leaving plenty of room 
    # for the 100 output tokens and the 576 image tokens under the 4096 limit.
    base_text = "This is a long context test to fill up the KV cache. "
    long_prompt = base_text * 350
    for i in range(n):
        question = PROMPT_QUESTIONS[i % len(PROMPT_QUESTIONS)]
        prompts.append({
            "prompt": f"USER: <image>\n{long_prompt}\n{question}\nASSISTANT:",
            "multi_modal_data": {"image": image},
        })
    return prompts


def get_vram_mib() -> float:
    """Best-effort device VRAM reading via pynvml → nvidia-smi fallback."""
    # Try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024 * 1024)
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            text=True,
        )
        return float(out.strip())
    except Exception:
        return 0.0


def is_oom_error(exc: Exception) -> bool:
    """Heuristic check for CUDA OOM or vLLM memory-related failures."""
    msg = str(exc).lower()
    patterns = [
        "out of memory",
        "oom",
        "cuda error",
        "cannot allocate",
        "cudamalloc",
        "no available memory",
        "insufficient",
    ]
    return any(p in msg for p in patterns)


def cuda_cleanup():
    """Force-free CUDA caches between engine runs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Single-mode benchmark (runs in-process)
# ---------------------------------------------------------------------------

def benchmark_mode(mode: str, args) -> dict:
    """
    Benchmark a single mode ("baseline" or "visionkv").
    Returns {"max_batch": int, "results": [...]}.
    """
    from vllm import LLM, SamplingParams

    image = create_dummy_image()
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    max_successful = 0
    results = []

    for batch_size in BATCH_SIZES:
        if batch_size > args.max_batch:
            break

        print(f"\n{'=' * 64}")
        print(f"  [{mode.upper()}]  batch_size = {batch_size}")
        print(f"{'=' * 64}")

        llm = None
        try:
            # ---- Create engine ----
            llm = LLM(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_mem_util,
                disable_log_stats=True,
                limit_mm_per_prompt={"image": 1},
                enforce_eager=args.enforce_eager,
            )

            # ---- Optionally install VisionKV ----
            if mode == "visionkv":
                from visionkv.live_integration import VisionKVPlugin
                from visionkv.policy import VisionKVPolicy
                policy = VisionKVPolicy(
                    hot_prefetch_block_count=args.hot_prefetch_block_count,
                    background_prefetch_remainder=True,
                )
                VisionKVPlugin(llm, policy=policy).install()
                print("  VisionKV plugin installed ✓")

            # ---- Build and run batch ----
            prompts = build_prompts(batch_size, image)
            vram_before = get_vram_mib()

            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling_params)
            elapsed = time.perf_counter() - t0

            vram_after = get_vram_mib()

            # ---- Collect metrics ----
            total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = total_out_tokens / elapsed if elapsed > 0 else 0
            avg_latency = elapsed / batch_size

            result = {
                "batch_size": batch_size,
                "status": "OK",
                "total_output_tokens": total_out_tokens,
                "wall_time_s": round(elapsed, 2),
                "throughput_tok_s": round(throughput, 1),
                "avg_latency_s": round(avg_latency, 3),
                "vram_before_mib": round(vram_before, 1),
                "vram_after_mib": round(vram_after, 1),
            }
            results.append(result)

            print(f"  ✓ SUCCESS  |  {total_out_tokens} tokens in {elapsed:.2f}s")
            print(f"             |  Throughput: {throughput:.1f} tok/s")
            print(f"             |  Avg latency/request: {avg_latency:.3f}s")
            print(f"             |  VRAM: {vram_before:.0f} → {vram_after:.0f} MiB")

            max_successful = batch_size

        except Exception as exc:
            if is_oom_error(exc):
                print(f"  ✗ OOM at batch_size={batch_size}")
                print(f"    Error: {exc}")
                results.append({
                    "batch_size": batch_size,
                    "status": "OOM",
                    "error": str(exc)[:200],
                })
                break
            else:
                print(f"  ✗ ERROR at batch_size={batch_size}")
                print(f"    {traceback.format_exc()}")
                results.append({
                    "batch_size": batch_size,
                    "status": "ERROR",
                    "error": str(exc)[:200],
                })
                # Non-OOM errors may be transient; keep trying higher sizes
                continue

        finally:
            # ---- Teardown engine ----
            if llm is not None:
                try:
                    del llm
                except Exception:
                    pass
            cuda_cleanup()
            # Small sleep to let GPU memory settle
            time.sleep(2)

    return {"max_batch": max_successful, "results": results}


# ---------------------------------------------------------------------------
# Subprocess-isolated runner
# ---------------------------------------------------------------------------

def run_mode_in_subprocess(mode: str, args) -> dict:
    """
    Run benchmark_mode() in an isolated subprocess to guarantee clean
    CUDA context between baseline and VisionKV runs.
    """
    # Serialize args to pass via environment
    env = os.environ.copy()
    env["_VISIONKV_BENCH_MODE"] = mode
    env["_VISIONKV_BENCH_ARGS"] = json.dumps({
        "model": args.model,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_mem_util": args.gpu_mem_util,
        "max_batch": args.max_batch,
        "hot_prefetch_block_count": args.hot_prefetch_block_count,
        "enforce_eager": args.enforce_eager,
    })

    result = subprocess.run(
        [sys.executable, __file__, "--_subprocess"],
        env=env,
        capture_output=False,  # Let output stream to terminal
    )

    # Read result from temp file
    result_path = f"/tmp/_visionkv_bench_{mode}.json"
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    else:
        return {"max_batch": 0, "results": [], "error": f"subprocess exit code {result.returncode}"}


def _subprocess_main():
    """Entry point when invoked as a subprocess."""
    mode = os.environ["_VISIONKV_BENCH_MODE"]
    raw_args = json.loads(os.environ["_VISIONKV_BENCH_ARGS"])

    # Reconstruct args namespace
    args = argparse.Namespace(**raw_args)

    result = benchmark_mode(mode, args)

    # Write result to temp file for parent
    result_path = f"/tmp/_visionkv_bench_{mode}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(baseline: dict, visionkv: dict):
    """Print the final comparison report."""
    b_max = baseline["max_batch"]
    v_max = visionkv["max_batch"]

    if b_max > 0:
        improvement = ((v_max - b_max) / b_max) * 100
        ratio = v_max / b_max
    else:
        improvement = float("inf")
        ratio = float("inf")

    separator = "=" * 64
    print(f"\n\n{separator}")
    print("  VisionKV Concurrency Benchmark — RESULTS")
    print(separator)
    print()
    print(f"  Standard vLLM Max Concurrent Users:  {b_max}")
    print(f"  VisionKV    Max Concurrent Users:     {v_max}")
    print()

    if b_max > 0 and v_max > 0:
        print(f"  Improvement: {ratio:.1f}× ({improvement:+.0f}%)")
    elif b_max == 0:
        print("  ⚠  Baseline could not complete any batch size.")
        print("     Try increasing --gpu-mem-util or reducing --max-batch.")
    print()

    # Detail tables
    for label, data in [("BASELINE", baseline), ("VISIONKV", visionkv)]:
        print(f"  ── {label} Detail ──")
        for r in data.get("results", []):
            status = r["status"]
            bs = r["batch_size"]
            if status == "OK":
                print(f"    batch={bs:>3}  ✓  {r['throughput_tok_s']:>7.1f} tok/s"
                      f"  latency={r['avg_latency_s']:.3f}s"
                      f"  VRAM={r['vram_after_mib']:.0f} MiB")
            else:
                print(f"    batch={bs:>3}  ✗  {status}")
        print()

    print(separator)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VisionKV Concurrency Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Quick test with tight memory
              python3 benchmark_concurrency.py --gpu-mem-util 0.40

              # Full sweep on 80 GB GPU
              python3 benchmark_concurrency.py --gpu-mem-util 0.50 --max-batch 40

              # Use eager mode (no CUDA graphs, less fixed overhead)
              python3 benchmark_concurrency.py --enforce-eager --gpu-mem-util 0.35
        """),
    )
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf",
                        help="HuggingFace model ID")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        dest="max_model_len",
                        help="Maximum sequence length")
    parser.add_argument("--max-tokens", type=int, default=100,
                        dest="max_tokens",
                        help="Output tokens per request")
    parser.add_argument("--gpu-mem-util", type=float, default=0.25,
                        dest="gpu_mem_util",
                        help="GPU memory utilization (lower = tighter KV budget)")
    parser.add_argument("--max-batch", type=int, default=8,
                        dest="max_batch",
                        help="Largest batch size to attempt")
    parser.add_argument("--hot-prefetch-block-count", type=int, default=2,
                        dest="hot_prefetch_block_count",
                        help="VisionKV hot-set prefetch block count")
    parser.add_argument("--enforce-eager", action="store_true",
                        dest="enforce_eager",
                        help="Disable CUDA graphs (reduces fixed memory)")
    parser.add_argument("--no-subprocess", action="store_true",
                        dest="no_subprocess",
                        help="Run both modes in-process (faster, but shared CUDA ctx)")
    parser.add_argument("--_subprocess", action="store_true",
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ---- Subprocess entry point ----
    if args._subprocess:
        _subprocess_main()
        return

    # ---- Print header ----
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         VisionKV Concurrency Benchmark                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Model:              {args.model}")
    print(f"  Max model len:      {args.max_model_len}")
    print(f"  Output tokens:      {args.max_tokens}")
    print(f"  GPU mem util:       {args.gpu_mem_util}")
    print(f"  Max batch size:     {args.max_batch}")
    print(f"  Batch sizes:        {[b for b in BATCH_SIZES if b <= args.max_batch]}")
    print(f"  Enforce eager:      {args.enforce_eager}")
    print(f"  Subprocess isolate: {not args.no_subprocess}")
    print()

    if args.no_subprocess:
        # ---- In-process mode ----
        print("\n" + "▸" * 64)
        print("  Phase 1: BASELINE (no VisionKV)")
        print("▸" * 64)
        baseline = benchmark_mode("baseline", args)
        cuda_cleanup()
        time.sleep(3)

        print("\n" + "▸" * 64)
        print("  Phase 2: VISIONKV (eviction active)")
        print("▸" * 64)
        visionkv = benchmark_mode("visionkv", args)

    else:
        # ---- Subprocess-isolated mode (default) ----
        print("\n" + "▸" * 64)
        print("  Phase 1: BASELINE (no VisionKV)  [subprocess]")
        print("▸" * 64)
        baseline = run_mode_in_subprocess("baseline", args)

        print("\n" + "▸" * 64)
        print("  Phase 2: VISIONKV (eviction active)  [subprocess]")
        print("▸" * 64)
        visionkv = run_mode_in_subprocess("visionkv", args)

    # ---- Final report ----
    print_report(baseline, visionkv)

    # Save raw data
    raw_output = {
        "config": {
            "model": args.model,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "gpu_mem_util": args.gpu_mem_util,
            "max_batch": args.max_batch,
        },
        "baseline": baseline,
        "visionkv": visionkv,
    }
    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(raw_output, f, indent=2)
    print(f"  Raw results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
