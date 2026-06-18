"""PyTorch prototype for real tensor allocation and transfer timing.

This is the "engine on a stand" stage for VisionKV:
- allocate real tensors for vision/text KV blocks,
- move vision tensors between accelerator memory and CPU memory,
- use non-blocking copies when CUDA is available,
- measure launch/completion timing so we can reason about overlap.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover - exercised by local environment
    torch = None


FLOAT32_BYTES = 4


def torch_available() -> bool:
    return torch is not None


def megabytes_to_numel(size_mb: int, bytes_per_element: int = FLOAT32_BYTES) -> int:
    return (size_mb * 1024 * 1024) // bytes_per_element


def format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f}MB"


def should_use_non_blocking_copy(
    source: str,
    destination: str,
    supports_true_async_copy: bool,
) -> bool:
    """Return whether this transfer path can use a non-blocking copy."""

    if not supports_true_async_copy:
        return False
    return {source, destination} == {"cpu", "cuda"}


def resolve_prefetch_block_count(total_blocks: int, requested_blocks: Optional[int]) -> int:
    """Clamp the requested hot-set size to the available number of blocks."""

    if requested_blocks is None:
        return total_blocks
    if requested_blocks <= 0:
        raise ValueError("prefetch_block_count must be positive when provided.")
    return min(total_blocks, requested_blocks)


@dataclass
class PrototypeConfig:
    num_vision_blocks: int = 10
    vision_block_mb: int = 256
    num_text_blocks: int = 50
    text_block_mb: int = 64
    text_eviction_threshold: int = 20
    prefetch_block_count: Optional[int] = None
    background_prefetch_remainder: bool = True
    overlap_matmul_dim: int = 1024
    preferred_device: str = "auto"


@dataclass
class RuntimeDevices:
    accelerator: str
    supports_true_async_copy: bool


@dataclass
class TensorBlock:
    logical_id: int
    modality: str
    size_mb: int
    location: str
    tensor: "torch.Tensor"
    cpu_staging_tensor: Optional["torch.Tensor"] = None

    @property
    def size_bytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


@dataclass
class TransferReport:
    direction: str
    total_bytes: int
    launch_ms: float
    total_ms: float
    overlapped_compute_ms: float
    used_non_blocking: bool
    blocks_moved: int
    used_staging_pool: bool


@dataclass
class PrefetchBudgetRecommendation:
    latency_budget_ms: float
    recommended_block_count: int
    recommended_bytes: int
    recommended_total_ms: float


@dataclass
class PrefetchSweepResult:
    reports: List[TransferReport]
    recommendation: PrefetchBudgetRecommendation


def recommend_prefetch_block_count(
    reports: List[TransferReport],
    latency_budget_ms: float,
) -> PrefetchBudgetRecommendation:
    """Pick the largest hot-set that stays within the flashback budget."""

    if latency_budget_ms <= 0:
        raise ValueError("latency_budget_ms must be positive.")
    if not reports:
        raise ValueError("At least one transfer report is required.")

    sorted_reports = sorted(reports, key=lambda report: report.blocks_moved)
    eligible_reports = [
        report for report in sorted_reports if report.total_ms <= latency_budget_ms
    ]
    chosen = eligible_reports[-1] if eligible_reports else sorted_reports[0]
    return PrefetchBudgetRecommendation(
        latency_budget_ms=latency_budget_ms,
        recommended_block_count=chosen.blocks_moved,
        recommended_bytes=chosen.total_bytes,
        recommended_total_ms=chosen.total_ms,
    )


def parse_int_csv(raw_value: str) -> List[int]:
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


class TorchVisionKVPrototype:
    """Real tensor transfer prototype for the next VisionKV stage."""

    def __init__(self, config: Optional[PrototypeConfig] = None) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is not installed. Install torch to run the prototype."
            )

        self.config = config or PrototypeConfig()
        self.devices = self._select_runtime_devices(self.config.preferred_device)
        self.transfer_stream = (
            torch.cuda.Stream() if self.devices.accelerator == "cuda" else None
        )
        self.blocks: Dict[int, TensorBlock] = {}
        self.next_logical_id = 0
        self.staging_pool_enabled = self.devices.accelerator == "cuda"

    @staticmethod
    def _select_runtime_devices(preferred_device: str) -> RuntimeDevices:
        if torch is None:
            raise RuntimeError("PyTorch is not installed.")

        if preferred_device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError(f"Unsupported preferred_device={preferred_device!r}")

        if preferred_device in {"auto", "cuda"} and torch.cuda.is_available():
            return RuntimeDevices(accelerator="cuda", supports_true_async_copy=True)

        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if preferred_device in {"auto", "mps"} and mps_available:
            return RuntimeDevices(accelerator="mps", supports_true_async_copy=False)

        if preferred_device in {"auto", "cpu"}:
            return RuntimeDevices(accelerator="cpu", supports_true_async_copy=False)

        raise RuntimeError(
            f"Requested device '{preferred_device}' is unavailable on this machine."
        )

    def allocate_block(self, modality: str, size_mb: int) -> TensorBlock:
        numel = megabytes_to_numel(size_mb)
        tensor = torch.empty(numel, dtype=torch.float32, device=self.devices.accelerator)
        cpu_staging_tensor = None
        if self.staging_pool_enabled and modality == "vision":
            cpu_staging_tensor = torch.empty_like(
                tensor,
                device="cpu",
                pin_memory=True,
            )
        block = TensorBlock(
            logical_id=self.next_logical_id,
            modality=modality,
            size_mb=size_mb,
            location=self.devices.accelerator,
            tensor=tensor,
            cpu_staging_tensor=cpu_staging_tensor,
        )
        self.blocks[block.logical_id] = block
        self.next_logical_id += 1
        return block

    def get_blocks(self, modality: Optional[str] = None, location: Optional[str] = None) -> List[TensorBlock]:
        return [
            block
            for block in self.blocks.values()
            if (modality is None or block.modality == modality)
            and (location is None or block.location == location)
        ]

    def total_bytes(self, location: Optional[str] = None) -> int:
        return sum(block.size_bytes for block in self.get_blocks(location=location))

    def offload_vision_blocks(self) -> TransferReport:
        return self._move_blocks(
            self.get_blocks(modality="vision", location=self.devices.accelerator),
            destination="cpu",
        )

    def prefetch_vision_blocks(self, block_limit: Optional[int] = None) -> TransferReport:
        cpu_vision_blocks = self.get_blocks(modality="vision", location="cpu")
        selected_blocks = cpu_vision_blocks[: resolve_prefetch_block_count(len(cpu_vision_blocks), block_limit)]
        return self._move_blocks(selected_blocks, destination=self.devices.accelerator)

    def run_demo(self) -> None:
        print("== VisionKV PyTorch Prototype ==")
        print(
            f"runtime_device={self.devices.accelerator} "
            f"true_async_copy={self.devices.supports_true_async_copy}"
        )

        print("\nPhase 1: allocate real vision tensors")
        for _ in range(self.config.num_vision_blocks):
            block = self.allocate_block("vision", self.config.vision_block_mb)
            print(
                f"allocated logical={block.logical_id:02d} modality=vision "
                f"size={block.size_mb}MB location={block.location}"
            )

        print("\nPhase 2: allocate text tensors until eviction threshold")
        offload_report: Optional[TransferReport] = None
        for step in range(1, self.config.num_text_blocks + 1):
            block = self.allocate_block("text", self.config.text_block_mb)
            print(
                f"decode={step:02d} text_logical={block.logical_id:02d} "
                f"gpu_bytes={format_bytes(self.total_bytes(self.devices.accelerator))}"
            )

            if step == self.config.text_eviction_threshold + 1:
                offload_report = self.offload_vision_blocks()
                print(self._format_transfer_report(offload_report))

        hot_prefetch_count = resolve_prefetch_block_count(
            total_blocks=len(self.get_blocks(modality="vision", location="cpu")),
            requested_blocks=self.config.prefetch_block_count,
        )
        print(
            "\nPhase 3: simulate a follow-up question about the image "
            f"(hot prefetch blocks={hot_prefetch_count})"
        )
        prefetch_report = self.prefetch_vision_blocks(block_limit=self.config.prefetch_block_count)
        print(self._format_transfer_report(prefetch_report))

        remaining_cpu_blocks = self.get_blocks(modality="vision", location="cpu")
        if remaining_cpu_blocks and self.config.background_prefetch_remainder:
            print(
                "\nPhase 4: continue background restore of remaining vision blocks "
                f"(remaining_blocks={len(remaining_cpu_blocks)})"
            )
            background_report = self.prefetch_vision_blocks()
            print(self._format_transfer_report(background_report))

    def run_until_prefetch(self) -> TransferReport:
        for _ in range(self.config.num_vision_blocks):
            self.allocate_block("vision", self.config.vision_block_mb)

        for step in range(1, self.config.num_text_blocks + 1):
            self.allocate_block("text", self.config.text_block_mb)
            if step == self.config.text_eviction_threshold + 1:
                self.offload_vision_blocks()

        return self.prefetch_vision_blocks(block_limit=self.config.prefetch_block_count)

    def _move_blocks(self, blocks: List[TensorBlock], destination: str) -> TransferReport:
        if not blocks:
            return TransferReport(
                direction=f"noop->{destination}",
                total_bytes=0,
                launch_ms=0.0,
                total_ms=0.0,
                overlapped_compute_ms=0.0,
                used_non_blocking=False,
                blocks_moved=0,
                used_staging_pool=False,
            )

        source = blocks[0].location
        non_blocking = should_use_non_blocking_copy(
            source=source,
            destination=destination,
            supports_true_async_copy=self.devices.supports_true_async_copy,
        )
        start_time = time.perf_counter()
        transferred: List[tuple[TensorBlock, "torch.Tensor"]] = []
        used_staging_pool = False

        if self.transfer_stream is not None and non_blocking:
            with torch.cuda.stream(self.transfer_stream):
                for block in blocks:
                    destination_tensor = self._get_destination_tensor(
                        block,
                        destination,
                        non_blocking,
                    )
                    if destination == "cpu" and destination_tensor is block.cpu_staging_tensor:
                        used_staging_pool = True
                    transferred.append(
                        (
                            block,
                            destination_tensor,
                        )
                    )
        else:
            for block in blocks:
                transferred.append((block, self._copy_tensor(block.tensor, destination, non_blocking)))

        launch_ms = (time.perf_counter() - start_time) * 1000
        overlapped_compute_ms = self._run_overlap_probe()
        self._synchronize_transfers()

        for block, new_tensor in transferred:
            block.tensor = new_tensor
            block.location = destination

        total_ms = (time.perf_counter() - start_time) * 1000
        return TransferReport(
            direction=f"{source}->{destination}",
            total_bytes=sum(block.size_bytes for block in blocks),
            launch_ms=launch_ms,
            total_ms=total_ms,
            overlapped_compute_ms=overlapped_compute_ms,
            used_non_blocking=non_blocking,
            blocks_moved=len(blocks),
            used_staging_pool=used_staging_pool,
        )

    def _get_destination_tensor(
        self,
        block: TensorBlock,
        destination: str,
        non_blocking: bool,
    ) -> "torch.Tensor":
        if destination == "cpu" and block.cpu_staging_tensor is not None:
            block.cpu_staging_tensor.copy_(block.tensor, non_blocking=non_blocking)
            return block.cpu_staging_tensor

        return self._copy_tensor(block.tensor, destination, non_blocking)

    def _copy_tensor(
        self, tensor: "torch.Tensor", destination: str, non_blocking: bool
    ) -> "torch.Tensor":
        if destination == "cpu":
            if self.devices.supports_true_async_copy:
                target = torch.empty_like(tensor, device="cpu", pin_memory=True)
                target.copy_(tensor, non_blocking=non_blocking)
                return target
            return tensor.to("cpu")

        if destination == "cuda":
            target = torch.empty_like(tensor, device="cuda")
            target.copy_(tensor, non_blocking=non_blocking)
            return target

        if destination == "mps":
            return tensor.to("mps")

        if destination == "cpu":
            return tensor.to("cpu")

        raise ValueError(f"Unsupported destination={destination!r}")

    def _run_overlap_probe(self) -> float:
        """Launch some work to see whether the transfer path blocks immediately."""

        start = time.perf_counter()
        dim = self.config.overlap_matmul_dim

        if self.devices.accelerator == "cuda":
            probe_a = torch.randn((dim, dim), device="cuda")
            probe_b = torch.randn((dim, dim), device="cuda")
            _ = probe_a @ probe_b
            torch.cuda.current_stream().synchronize()
        elif self.devices.accelerator == "mps":
            probe_a = torch.randn((dim, dim), device="mps")
            probe_b = torch.randn((dim, dim), device="mps")
            _ = probe_a @ probe_b
            if hasattr(torch, "mps"):
                torch.mps.synchronize()
        else:
            probe_a = torch.randn((dim, dim))
            probe_b = torch.randn((dim, dim))
            _ = probe_a @ probe_b

        return (time.perf_counter() - start) * 1000

    def _synchronize_transfers(self) -> None:
        if self.transfer_stream is not None:
            self.transfer_stream.synchronize()
        elif self.devices.accelerator == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    @staticmethod
    def _format_transfer_report(report: TransferReport) -> str:
        return (
            f"{report.direction} blocks={report.blocks_moved} "
            f"bytes={format_bytes(report.total_bytes)} "
            f"launch_ms={report.launch_ms:.2f} "
            f"total_ms={report.total_ms:.2f} "
            f"overlap_probe_ms={report.overlapped_compute_ms:.2f} "
            f"non_blocking={report.used_non_blocking} "
            f"staging_pool={report.used_staging_pool}"
        )


def run_prefetch_sweep(
    config: PrototypeConfig,
    block_counts: List[int],
    latency_budget_ms: float,
) -> PrefetchSweepResult:
    reports: List[TransferReport] = []
    for block_count in block_counts:
        sweep_config = PrototypeConfig(
            num_vision_blocks=config.num_vision_blocks,
            vision_block_mb=config.vision_block_mb,
            num_text_blocks=config.num_text_blocks,
            text_block_mb=config.text_block_mb,
            text_eviction_threshold=config.text_eviction_threshold,
            prefetch_block_count=block_count,
            background_prefetch_remainder=False,
            overlap_matmul_dim=config.overlap_matmul_dim,
            preferred_device=config.preferred_device,
        )
        prototype = TorchVisionKVPrototype(sweep_config)
        reports.append(prototype.run_until_prefetch())

    return PrefetchSweepResult(
        reports=reports,
        recommendation=recommend_prefetch_block_count(reports, latency_budget_ms),
    )


def format_prefetch_budget_recommendation(
    recommendation: PrefetchBudgetRecommendation,
) -> str:
    return (
        f"budget_ms={recommendation.latency_budget_ms:.2f} "
        f"recommended_blocks={recommendation.recommended_block_count} "
        f"recommended_bytes={format_bytes(recommendation.recommended_bytes)} "
        f"expected_total_ms={recommendation.recommended_total_ms:.2f}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VisionKV PyTorch prototype.")
    parser.add_argument("--num-vision-blocks", type=int, default=10)
    parser.add_argument("--vision-block-mb", type=int, default=256)
    parser.add_argument("--num-text-blocks", type=int, default=50)
    parser.add_argument("--text-block-mb", type=int, default=64)
    parser.add_argument("--text-eviction-threshold", type=int, default=20)
    parser.add_argument("--prefetch-block-count", type=int)
    parser.add_argument(
        "--no-background-prefetch-remainder",
        action="store_true",
        help="Skip the follow-on restore of blocks not included in the initial hot prefetch.",
    )
    parser.add_argument(
        "--prefetch-sweep-counts",
        help="Comma-separated hot-set sizes to benchmark, for example '1,2,4'.",
    )
    parser.add_argument(
        "--flashback-budget-ms",
        type=float,
        default=50.0,
        help="Latency budget used when recommending a hot-set size from a sweep.",
    )
    parser.add_argument("--overlap-matmul-dim", type=int, default=1024)
    parser.add_argument("--preferred-device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    if torch is None:
        print("PyTorch is not installed. Install torch to run this prototype.")
        return 1

    args = build_arg_parser().parse_args(argv)
    config = PrototypeConfig(
        num_vision_blocks=args.num_vision_blocks,
        vision_block_mb=args.vision_block_mb,
        num_text_blocks=args.num_text_blocks,
        text_block_mb=args.text_block_mb,
        text_eviction_threshold=args.text_eviction_threshold,
        prefetch_block_count=args.prefetch_block_count,
        background_prefetch_remainder=not args.no_background_prefetch_remainder,
        overlap_matmul_dim=args.overlap_matmul_dim,
        preferred_device=args.preferred_device,
    )
    if args.prefetch_sweep_counts:
        sweep = run_prefetch_sweep(
            config=config,
            block_counts=parse_int_csv(args.prefetch_sweep_counts),
            latency_budget_ms=args.flashback_budget_ms,
        )
        print("== VisionKV Prefetch Sweep ==")
        for report in sweep.reports:
            print(TorchVisionKVPrototype._format_transfer_report(report))
        print(format_prefetch_budget_recommendation(sweep.recommendation))
        return 0

    prototype = TorchVisionKVPrototype(config)
    prototype.run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
