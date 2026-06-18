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


@dataclass
class PrototypeConfig:
    num_vision_blocks: int = 10
    vision_block_mb: int = 256
    num_text_blocks: int = 50
    text_block_mb: int = 64
    text_eviction_threshold: int = 20
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
        block = TensorBlock(
            logical_id=self.next_logical_id,
            modality=modality,
            size_mb=size_mb,
            location=self.devices.accelerator,
            tensor=tensor,
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

    def prefetch_vision_blocks(self) -> TransferReport:
        return self._move_blocks(
            self.get_blocks(modality="vision", location="cpu"),
            destination=self.devices.accelerator,
        )

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

        print("\nPhase 3: simulate a follow-up question about the image")
        prefetch_report = self.prefetch_vision_blocks()
        print(self._format_transfer_report(prefetch_report))

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
            )

        non_blocking = self.devices.supports_true_async_copy and destination == "cpu"
        start_time = time.perf_counter()
        transferred: List[tuple[TensorBlock, "torch.Tensor"]] = []

        if self.transfer_stream is not None and destination == "cpu":
            with torch.cuda.stream(self.transfer_stream):
                for block in blocks:
                    cpu_tensor = torch.empty_like(block.tensor, device="cpu", pin_memory=True)
                    cpu_tensor.copy_(block.tensor, non_blocking=True)
                    transferred.append((block, cpu_tensor))
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
            direction=f"{blocks[0].location}->{destination}",
            total_bytes=sum(block.size_bytes for block in blocks),
            launch_ms=launch_ms,
            total_ms=total_ms,
            overlapped_compute_ms=overlapped_compute_ms,
            used_non_blocking=non_blocking,
            blocks_moved=len(blocks),
        )

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
            f"non_blocking={report.used_non_blocking}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VisionKV PyTorch prototype.")
    parser.add_argument("--num-vision-blocks", type=int, default=10)
    parser.add_argument("--vision-block-mb", type=int, default=256)
    parser.add_argument("--num-text-blocks", type=int, default=50)
    parser.add_argument("--text-block-mb", type=int, default=64)
    parser.add_argument("--text-eviction-threshold", type=int, default=20)
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
        overlap_matmul_dim=args.overlap_matmul_dim,
        preferred_device=args.preferred_device,
    )
    prototype = TorchVisionKVPrototype(config)
    prototype.run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
