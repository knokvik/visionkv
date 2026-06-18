"""Run the live VisionKV monkey-patch against a pip-installed vLLM runtime."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm import LLM, SamplingParams

from visionkv.live_integration import VisionKVPlugin
from visionkv.policy import VisionKVPolicy

try:
    import pynvml
except ImportError:  # pragma: no cover - depends on remote GPU env
    pynvml = None


LOGGER = logging.getLogger("run_visionkv_vllm")


def build_dummy_image(size: int) -> Image.Image:
    grid = np.indices((size, size)).sum(axis=0) % 2
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = np.where(grid == 0, 32, 220)
    image[..., 1] = np.where(grid == 0, 160, 64)
    image[..., 2] = np.where(grid == 0, 240, 32)
    return Image.fromarray(image, mode="RGB")


def format_vram_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def build_llava_prompt(question: str) -> str:
    return f"USER: <image>\n{question}\nASSISTANT:"


def print_snapshot(label: str, payload: Any) -> None:
    print(label)
    print(json.dumps(payload, indent=2, sort_keys=True))


class DeviceMemoryMonitor:
    """Poll total device VRAM via NVML from outside the vLLM worker process."""

    def __init__(self, device_index: int, interval_s: float = 0.02) -> None:
        self.device_index = device_index
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_bytes = 0
        self._peak_bytes = 0
        self._handle: Any | None = None

    @property
    def available(self) -> bool:
        return pynvml is not None

    @property
    def baseline_bytes(self) -> int:
        return self._baseline_bytes

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    def reset_baseline(self) -> int:
        current = self.read_used_bytes()
        self._baseline_bytes = current
        self._peak_bytes = max(self._peak_bytes, current)
        return current

    def initialize(self) -> None:
        if pynvml is None:
            return
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._baseline_bytes = self.read_used_bytes()
        self._peak_bytes = self._baseline_bytes

    def shutdown(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:
            LOGGER.debug("NVML shutdown failed", exc_info=True)

    def read_used_bytes(self) -> int:
        if pynvml is None or self._handle is None:
            return 0
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return int(info.used)

    def start(self) -> None:
        if pynvml is None or self._handle is None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 10))
        self._thread = None
        self._peak_bytes = max(self._peak_bytes, self.read_used_bytes())

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._peak_bytes = max(self._peak_bytes, self.read_used_bytes())
            except Exception:
                LOGGER.debug("NVML poll failed", exc_info=True)
                return
            time.sleep(self.interval_s)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VisionKV against vLLM LLaVA.")
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--followup-max-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--hot-prefetch-block-count", type=int, default=2)
    parser.add_argument("--flashback-budget-ms", type=float, default=50.0)
    parser.add_argument("--nvml-poll-ms", type=float, default=20.0)
    parser.add_argument(
        "--disable-background-prefetch-remainder",
        action="store_true",
        help="Only prefetch the hot set on follow-up prompts.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-visible GPU.")

    device_index = torch.cuda.current_device()
    memory_monitor = DeviceMemoryMonitor(
        device_index=device_index,
        interval_s=max(args.nvml_poll_ms, 1.0) / 1000.0,
    )
    memory_monitor.initialize()

    image = build_dummy_image(args.image_size)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
    )

    baseline_vram = memory_monitor.reset_baseline() or memory_monitor.baseline_bytes
    print(f"Baseline VRAM: {format_vram_mib(baseline_vram)}")

    policy = VisionKVPolicy(
        hot_prefetch_block_count=args.hot_prefetch_block_count,
        flashback_budget_ms=args.flashback_budget_ms,
        background_prefetch_remainder=not args.disable_background_prefetch_remainder,
    )
    plugin = VisionKVPlugin(llm, policy=policy).install()

    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    primary_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    primary_prompt = build_llava_prompt(
        "Describe this synthetic checkerboard image in detail, including the "
        "color pattern, repeated structure, and anything near the center."
    )
    try:
        memory_monitor.start()
        primary_outputs = llm.generate(
            {
                "prompt": primary_prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=primary_sampling_params,
        )
        torch.cuda.synchronize()
        memory_monitor.stop()

        peak_vram = memory_monitor.peak_bytes or memory_monitor.read_used_bytes()
        print(f"Peak VRAM during generation: {format_vram_mib(peak_vram)}")
        print("Generated text:")
        print(primary_outputs[0].outputs[0].text)

        followup_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.followup_max_tokens,
        )
        followup_prompt = build_llava_prompt(
            "Now answer again in one short sentence and focus only on what is visible in the center."
        )
        memory_monitor.start()
        followup_outputs = llm.generate(
            {
                "prompt": followup_prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=followup_sampling_params,
        )
        torch.cuda.synchronize()
        memory_monitor.stop()

        print("Follow-up generated text:")
        print(followup_outputs[0].outputs[0].text)
        print_snapshot("VisionKV snapshot:", plugin.snapshot())
        print(
            "Peak device VRAM across both requests: "
            f"{format_vram_mib(memory_monitor.peak_bytes)}"
        )
        return 0
    finally:
        memory_monitor.stop()
        memory_monitor.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
