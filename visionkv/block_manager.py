"""Mock block management primitives inspired by vLLM's block table."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PhysicalBlock:
    """Represents a GPU-resident physical KV block."""

    physical_id: int
    tensor_size_mb: int
    owner_logical_id: Optional[int] = None


@dataclass
class LogicalBlock:
    """Represents a logical KV block tracked across GPU/CPU tiers."""

    logical_id: int
    tensor_size_mb: int
    modality: str
    location: str = "gpu"
    physical_block_id: Optional[int] = None
    cpu_slot_id: Optional[str] = None


class MockBlockSpaceManager:
    """Owns logical blocks and a reusable pool of physical GPU blocks."""

    def __init__(self) -> None:
        self.block_table: Dict[int, LogicalBlock] = {}
        self.physical_blocks: Dict[int, PhysicalBlock] = {}
        self.free_physical_ids: List[int] = []
        self.next_logical_id = 0
        self.next_physical_id = 0

    def allocate_block(self, modality: str, tensor_size_mb: int) -> LogicalBlock:
        physical_block = self._acquire_physical_block(tensor_size_mb)
        logical_block = LogicalBlock(
            logical_id=self.next_logical_id,
            tensor_size_mb=tensor_size_mb,
            modality=modality,
            location="gpu",
            physical_block_id=physical_block.physical_id,
        )
        physical_block.owner_logical_id = logical_block.logical_id
        self.block_table[logical_block.logical_id] = logical_block
        self.next_logical_id += 1
        return logical_block

    def get_blocks(
        self, modality: Optional[str] = None, location: Optional[str] = None
    ) -> List[LogicalBlock]:
        return [
            block
            for block in self.block_table.values()
            if (modality is None or block.modality == modality)
            and (location is None or block.location == location)
        ]

    def offload_blocks(self, logical_ids: List[int]) -> int:
        freed_mb = 0
        for logical_id in logical_ids:
            block = self.block_table[logical_id]
            if block.location != "gpu" or block.physical_block_id is None:
                continue

            physical_block = self.physical_blocks[block.physical_block_id]
            physical_block.owner_logical_id = None
            self.free_physical_ids.append(physical_block.physical_id)

            block.location = "cpu"
            block.cpu_slot_id = f"cpu:{block.logical_id}"
            block.physical_block_id = None
            freed_mb += block.tensor_size_mb
        return freed_mb

    def prefetch_blocks(self, logical_ids: List[int]) -> int:
        restored_mb = 0
        for logical_id in logical_ids:
            block = self.block_table[logical_id]
            if block.location != "cpu":
                continue

            physical_block = self._acquire_physical_block(block.tensor_size_mb)
            physical_block.owner_logical_id = block.logical_id

            block.location = "gpu"
            block.physical_block_id = physical_block.physical_id
            block.cpu_slot_id = None
            restored_mb += block.tensor_size_mb
        return restored_mb

    def gpu_memory_mb(self) -> int:
        return sum(
            block.tensor_size_mb
            for block in self.block_table.values()
            if block.location == "gpu"
        )

    def cpu_memory_mb(self) -> int:
        return sum(
            block.tensor_size_mb
            for block in self.block_table.values()
            if block.location == "cpu"
        )

    def text_block_count(self) -> int:
        return len(self.get_blocks(modality="text"))

    def free_gpu_block_count(self) -> int:
        return sum(1 for block in self.physical_blocks.values() if block.owner_logical_id is None)

    def summary(self) -> str:
        return (
            f"gpu_blocks={len(self.get_blocks(location='gpu'))} "
            f"gpu_memory={self.gpu_memory_mb()}MB | "
            f"cpu_blocks={len(self.get_blocks(location='cpu'))} "
            f"cpu_memory={self.cpu_memory_mb()}MB | "
            f"free_gpu_slots={self.free_gpu_block_count()}"
        )

    def _acquire_physical_block(self, tensor_size_mb: int) -> PhysicalBlock:
        if self.free_physical_ids:
            physical_id = self.free_physical_ids.pop(0)
            physical_block = self.physical_blocks[physical_id]
            physical_block.tensor_size_mb = tensor_size_mb
            return physical_block

        physical_block = PhysicalBlock(
            physical_id=self.next_physical_id,
            tensor_size_mb=tensor_size_mb,
        )
        self.physical_blocks[physical_block.physical_id] = physical_block
        self.next_physical_id += 1
        return physical_block
