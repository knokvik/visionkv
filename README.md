# VisionKV

**Offload vision KV cache blocks to CPU RAM during text generation — serve more concurrent VLM users on the same GPU.**

VisionKV is a vLLM plugin for Vision-Language Models (VLMs) that dynamically evicts vision key-value (KV) cache blocks to CPU memory once the model stops attending to them during text generation. When a follow-up question arrives, blocks are prefetched back with a budgeted hot-set strategy, keeping flashback latency under a configurable threshold.

## The Problem

In a VLM like LLaVA, the vision encoder injects hundreds of KV-cache blocks at prefill time. During subsequent text-only generation, attention to those vision tokens decays rapidly — yet vLLM keeps every block in GPU VRAM. This wastes precious memory that could serve additional concurrent requests, leading to premature OOM crashes.

## How VisionKV Works

1. **Tag** — When a multimodal prompt is preprocessed, VisionKV identifies which KV-cache blocks belong to vision tokens.
2. **Evict** — After a configurable number of text-generation tokens, vision blocks are asynchronously offloaded to pinned CPU memory, freeing GPU VRAM for new requests.
3. **Prefetch** — If a follow-up question about the image arrives, a hot-set of vision blocks is restored to GPU within a latency budget (default 50 ms). The remainder streams back in the background.
4. **Repeat** — The cycle continues for multi-turn conversations, keeping VRAM usage low while preserving image understanding.

### Architecture

```
┌────────────────────────────────────────────┐
│              vLLM Engine (V1)              │
│  ┌──────────────┐   ┌───────────────────┐  │
│  │ InputProcessor│──▶│   Worker          │  │
│  │  (tag vision) │   │  execute_model()  │  │
│  └──────────────┘   │  (offload/prefetch)│  │
│                     └───────┬───────────┘  │
│                             │              │
│  ┌──────────────────────────▼───────────┐  │
│  │        VisionKVPlugin               │  │
│  │  ┌──────────────┐ ┌──────────────┐  │  │
│  │  │ MetadataStore│ │ Controller   │  │  │
│  │  └──────────────┘ └──────┬───────┘  │  │
│  │                         │           │  │
│  │  ┌──────────────────────▼────────┐  │  │
│  │  │   TensorOffloadManager       │  │  │
│  │  │  GPU ←→ pinned CPU (async)   │  │  │
│  │  └──────────────────────────────┘  │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

## Project Structure

```
visionkv/
├── __init__.py              # Public API exports
├── policy.py                # VisionKVPolicy knobs (hot-set size, flashback budget)
├── pytorch_prototype.py     # Standalone tensor transfer prototype & prefetch sweep
├── block_manager.py         # Mock block space manager (for simulation)
├── controller.py            # Mock controller with cold/hot attention heuristics
├── simulation.py            # Runnable async simulation
├── vllm_adapter.py           # vLLM-facing hook surface (adapter layer)
├── integration_harness.py    # OpenAI-compatible API crash-test harness
├── benchmark_harness.py      # Baseline-vs-VisionKV comparison helpers
└── live_integration.py       # Live monkey-patch plugin for pip-installed vLLM

benchmark_concurrency.py      # Concurrency benchmark (main entry point)
run_visionkv_vllm.py          # Single-request live integration runner
visionkv_pytorch_prototype.py # CLI for the PyTorch prototype
visionkv_mock_simulation.py   # CLI for the mock simulation
visionkv_benchmark.py         # CLI for benchmark comparison helper
visionkv_integration_test.py  # CLI for integration harness
tests/                        # Unit tests
```

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x support (tested on A6000 48 GB)
- At least 16 GB of system RAM beyond the model's footprint

### Setup

```bash
# Clone the repository
git clone https://github.com/knokvik/visionkv.git
cd visionkv

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) For CUDA 12.8 with cuDNN:
# pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## Quick Start

### Run the Concurrency Benchmark

This compares the maximum number of concurrent users standard vLLM can handle vs. VisionKV:

```bash
# Tight memory — forces OOM at realistic batch sizes
python3 benchmark_concurrency.py --gpu-mem-util 0.25 --enforce-eager

# Relaxed memory — test on larger GPUs
python3 benchmark_concurrency.py --gpu-mem-util 0.40 --max-batch 40

# Full options
python3 benchmark_concurrency.py \
    --model llava-hf/llava-1.5-7b-hf \
    --max-model-len 4096 \
    --max-tokens 100 \
    --gpu-mem-util 0.30 \
    --max-batch 20 \
    --hot-prefetch-block-count 2 \
    --enforce-eager
```

Results are saved to `benchmark_results.json`.

### Run a Single-Request Live Test

```bash
python3 run_visionkv_vllm.py --model llava-hf/llava-1.5-7b-hf
```

This sends a two-turn conversation (initial question + follow-up) with VRAM telemetry.

### Run the PyTorch Prototype (No vLLM Required)

```bash
# Basic demo — shows offload, hot prefetch, and background restore
python3 visionkv_pytorch_prototype.py

# Prefetch sweep — find the optimal hot-set size within a latency budget
python3 visionkv_pytorch_prototype.py --prefetch-sweep-counts 1,2,4,6,8 --flashback-budget-ms 50.0

# On CPU (no GPU required)
python3 visionkv_pytorch_prototype.py --preferred-device cpu
```

### Run the Mock Simulation

```bash
python3 visionkv_mock_simulation.py
```

### Run Tests

```bash
python3 -m pytest tests/ -v
```

## Benchmark Flags (benchmark_concurrency.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `llava-hf/llava-1.5-7b-hf` | HuggingFace model ID |
| `--max-model-len` | `4096` | Maximum sequence length |
| `--max-tokens` | `100` | Output tokens per request |
| `--gpu-mem-util` | `0.25` | GPU memory utilization (lower = tighter KV budget, earlier OOM) |
| `--max-batch` | `8` | Largest batch size to attempt |
| `--hot-prefetch-block-count` | `2` | Number of vision blocks in the hot prefetch set |
| `--enforce-eager` | `false` | Disable CUDA graphs (reduces fixed memory overhead) |
| `--no-subprocess` | `false` | Run both modes in-process (faster but shared CUDA context) |

## Policy Configuration

The `VisionKVPolicy` dataclass controls the eviction/prefetch behavior:

```python
from visionkv.policy import VisionKVPolicy

policy = VisionKVPolicy(
    hot_prefetch_block_count=2,      # Blocks to eagerly restore on follow-up
    flashback_budget_ms=50.0,        # Max latency budget for hot prefetch
    background_prefetch_remainder=True,  # Stream remaining blocks in background
)
```

## How It Works Under the Hood

VisionKV intercepts vLLM's V1 engine at three points via monkey-patching:

1. **`InputProcessor.process_inputs`** — Tags vision token spans when a multimodal prompt is tokenized.
2. **`LLMEngine.add_request`** — Registers the request's vision block metadata and triggers prefetch for previously offloaded requests.
3. **`Worker.execute_model`** — Advances the offload/prefetch lifecycle at each decode step.

Physical tensor migration uses a dedicated CUDA stream with pinned CPU memory for zero-stall, non-blocking transfers.

## Tested Configuration

| Component | Version |
|-----------|---------|
| vLLM | 0.23.0 |
| PyTorch | 2.8.0 (CUDA 12.8) |
| Model | llava-hf/llava-1.5-7b-hf |
| GPU | NVIDIA A6000 (48 GB) |
| Python | 3.10+ |

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/knokvik/visionkv).
