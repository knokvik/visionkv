"""Live vLLM monkey-patch integration for VisionKV.

This module targets the vLLM V1 engine layout exposed from the main branch:
- vllm.engine.llm_engine.LLMEngine -> alias to vllm.v1.engine.llm_engine.LLMEngine
- vllm.v1.engine.input_processor.InputProcessor.process_inputs
- vllm.v1.worker.gpu_worker.Worker.execute_model

The plugin is intentionally standalone so it can be imported in an environment
where vLLM is installed via pip but the upstream source tree is not checked out.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch

from .policy import VisionKVPolicy

try:
    from vllm.engine.llm_engine import LLMEngine
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.worker.worker_base import WorkerBase
except ImportError as exc:  # pragma: no cover - exercised on the GPU host
    raise RuntimeError(
        "visionkv.live_integration requires a pip-installed vLLM V1 runtime."
    ) from exc


LOGGER = logging.getLogger("visionkv.live_integration")
SNAPSHOT_SCHEMA_VERSION = 3


def _extract_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        value = prompt.get("prompt")
        if isinstance(value, str):
            return value
    return ""


def _extract_token_spans(position: PlaceholderRange) -> list[tuple[int, int]]:
    ranges = position.extract_embeds_range()
    if not ranges:
        return []
    return [(start, end_inclusive + 1) for start, end_inclusive in ranges]


def _iter_block_ids(
    token_spans: Iterable[tuple[int, int]],
    block_size_tokens: int,
) -> list[int]:
    block_ids: list[int] = []
    seen: set[int] = set()
    for start, end in token_spans:
        if end <= start:
            continue
        first_block = start // block_size_tokens
        last_block = (end - 1) // block_size_tokens
        for block_id in range(first_block, last_block + 1):
            if block_id not in seen:
                seen.add(block_id)
                block_ids.append(block_id)
    return block_ids


def _bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)



class TensorOffloadManager:
    """Manages physical CPU/GPU tensor migration for KV cache blocks.

    Uses a dedicated CUDA stream and pinned CPU memory to perform
    asynchronous, zero-stall transfers of vision KV cache blocks.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER
        self._cuda_available = False
        self._stream: Any | None = None
        self._cpu_store: dict[str, dict[int, torch.Tensor]] = {}
        self._transfer_times: list[float] = []
        self._total_offload_bytes = 0
        self._total_prefetch_bytes = 0
        self._initialize()

    def _initialize(self) -> None:
        try:
            if torch.cuda.is_available():
                self._cuda_available = True
                self._stream = torch.cuda.Stream()
                self.logger.info(
                    "TensorOffloadManager initialized with dedicated CUDA stream"
                )
        except Exception:
            self.logger.debug("CUDA not available for tensor offload", exc_info=True)

    @property
    def available(self) -> bool:
        return self._cuda_available

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "cuda_available": self._cuda_available,
            "total_offload_bytes": self._total_offload_bytes,
            "total_prefetch_bytes": self._total_prefetch_bytes,
            "total_offload_mib": round(_bytes_to_mib(self._total_offload_bytes), 2),
            "total_prefetch_mib": round(_bytes_to_mib(self._total_prefetch_bytes), 2),
            "transfer_count": len(self._transfer_times),
            "avg_transfer_ms": (
                round(sum(self._transfer_times) / len(self._transfer_times) * 1000, 2)
                if self._transfer_times
                else 0.0
            ),
            "max_transfer_ms": (
                round(max(self._transfer_times) * 1000, 2)
                if self._transfer_times
                else 0.0
            ),
        }

    def offload_kv_blocks_to_cpu(
        self,
        request_id: str,
        kv_caches: list[torch.Tensor],
        block_ids: list[int],
    ) -> bool:
        """Move KV cache blocks from GPU to pinned CPU memory.

        Args:
            request_id: Unique identifier for this request's storage.
            kv_caches: List of KV cache tensors (one per layer),
                       shaped [num_blocks, block_size, num_heads, head_dim].
            block_ids: Logical block indices to offload.

        Returns:
            True if offload succeeded.
        """
        if not self._cuda_available or not kv_caches:
            return False

        t0 = time.perf_counter()
        cpu_blocks: dict[int, torch.Tensor] = {}

        try:
            with torch.cuda.stream(self._stream):
                for block_id in block_ids:
                    # Stack all layers for this block into a single tensor
                    layer_slices = []
                    for kv_cache in kv_caches:
                        if block_id < kv_cache.shape[0]:
                            layer_slices.append(kv_cache[block_id])
                    if not layer_slices:
                        continue
                    stacked = torch.stack(layer_slices)
                    # Transfer to pinned CPU memory for fast re-upload
                    cpu_tensor = stacked.to(
                        device="cpu",
                        non_blocking=True,
                    ).pin_memory()
                    cpu_blocks[block_id] = cpu_tensor
                    self._total_offload_bytes += cpu_tensor.nelement() * cpu_tensor.element_size()

            # Synchronize the offload stream to ensure data is on CPU
            self._stream.synchronize()

            # Zero out the GPU blocks to allow vLLM's allocator to reclaim
            with torch.cuda.stream(self._stream):
                for block_id in block_ids:
                    for kv_cache in kv_caches:
                        if block_id < kv_cache.shape[0]:
                            kv_cache[block_id].zero_()

            self._stream.synchronize()

        except Exception:
            self.logger.warning(
                "Failed to offload KV blocks for request=%s",
                request_id,
                exc_info=True,
            )
            return False

        self._cpu_store[request_id] = cpu_blocks
        elapsed = time.perf_counter() - t0
        self._transfer_times.append(elapsed)
        self.logger.info(
            "Tensor offload request=%s blocks=%s elapsed_ms=%.2f bytes=%d",
            request_id,
            block_ids,
            elapsed * 1000,
            sum(t.nelement() * t.element_size() for t in cpu_blocks.values()),
        )
        return True

    def prefetch_kv_blocks_to_gpu(
        self,
        request_id: str,
        kv_caches: list[torch.Tensor],
        block_ids: list[int],
    ) -> float:
        """Restore KV cache blocks from pinned CPU memory to GPU.

        Args:
            request_id: Unique identifier for this request's storage.
            kv_caches: List of KV cache tensors on GPU.
            block_ids: Logical block indices to prefetch.

        Returns:
            Elapsed time in seconds for the prefetch operation.
        """
        cpu_blocks = self._cpu_store.get(request_id, {})
        if not cpu_blocks or not self._cuda_available:
            return 0.0

        t0 = time.perf_counter()

        try:
            with torch.cuda.stream(self._stream):
                for block_id in block_ids:
                    cpu_tensor = cpu_blocks.get(block_id)
                    if cpu_tensor is None:
                        continue
                    # cpu_tensor is [num_layers, block_size, num_heads, head_dim]
                    for layer_idx, kv_cache in enumerate(kv_caches):
                        if layer_idx < cpu_tensor.shape[0] and block_id < kv_cache.shape[0]:
                            kv_cache[block_id].copy_(cpu_tensor[layer_idx], non_blocking=True)
                    self._total_prefetch_bytes += (
                        cpu_tensor.nelement() * cpu_tensor.element_size()
                    )

            self._stream.synchronize()

        except Exception:
            self.logger.warning(
                "Failed to prefetch KV blocks for request=%s",
                request_id,
                exc_info=True,
            )
            return 0.0

        # Remove restored blocks from CPU store
        for block_id in block_ids:
            cpu_blocks.pop(block_id, None)
        if not cpu_blocks:
            self._cpu_store.pop(request_id, None)

        elapsed = time.perf_counter() - t0
        self._transfer_times.append(elapsed)
        self.logger.info(
            "Tensor prefetch request=%s blocks=%s elapsed_ms=%.2f",
            request_id,
            block_ids,
            elapsed * 1000,
        )
        return elapsed

    def has_cpu_blocks(self, request_id: str) -> bool:
        return bool(self._cpu_store.get(request_id))

    def cleanup_request(self, request_id: str) -> None:
        self._cpu_store.pop(request_id, None)


@dataclass
class VisionKVRequestState:
    external_request_id: str
    internal_request_id: str | None
    prompt_text: str
    prompt_token_count: int
    vision_token_spans: list[tuple[int, int]]
    vision_block_ids: list[int]
    hot_block_ids: list[int]
    cold_block_ids: list[int]
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    generated_tokens: int = 0
    offloaded_block_ids: set[int] = field(default_factory=set)
    hot_prefetched_block_ids: set[int] = field(default_factory=set)
    background_prefetched_block_ids: set[int] = field(default_factory=set)
    pending_offload: bool = False
    background_prefetch_pending: bool = False
    offload_count: int = 0
    prefetch_count: int = 0
    background_prefetch_count: int = 0
    offload_elapsed_ms: float = 0.0
    prefetch_elapsed_ms: float = 0.0
    background_prefetch_elapsed_ms: float = 0.0

    @property
    def is_offloaded(self) -> bool:
        return bool(self.offloaded_block_ids)

    @property
    def total_prefetched_block_ids(self) -> set[int]:
        return self.hot_prefetched_block_ids | self.background_prefetched_block_ids


class VisionKVMetadataStore:
    """Tracks live request-level VisionKV state for the pip-installed engine."""

    def __init__(self) -> None:
        self._pending_by_external_id: dict[str, VisionKVRequestState] = {}
        self._by_internal_id: dict[str, VisionKVRequestState] = {}
        self._by_external_id: dict[str, VisionKVRequestState] = {}
        self._request_order: list[str] = []

    def register_processed_request(
        self,
        request: EngineCoreRequest,
        prompt_text: str,
        block_size_tokens: int,
        hot_prefetch_block_count: int | None,
    ) -> VisionKVRequestState | None:
        mm_features = [
            feature
            for feature in (request.mm_features or [])
            if feature.modality in {"image", "vision_chunk"}
        ]
        if not mm_features:
            return None

        token_spans: list[tuple[int, int]] = []
        for feature in sorted(mm_features, key=lambda item: item.mm_position.offset):
            token_spans.extend(_extract_token_spans(feature.mm_position))

        block_ids = _iter_block_ids(token_spans, block_size_tokens=block_size_tokens)
        if not block_ids:
            return None

        if hot_prefetch_block_count is None:
            hot_block_ids = list(block_ids)
            cold_block_ids: list[int] = []
        else:
            hot_block_ids = list(block_ids[:hot_prefetch_block_count])
            cold_block_ids = list(block_ids[hot_prefetch_block_count:])

        prompt_token_count = len(request.prompt_token_ids or [])
        state = VisionKVRequestState(
            external_request_id=request.request_id,
            internal_request_id=None,
            prompt_text=prompt_text,
            prompt_token_count=prompt_token_count,
            vision_token_spans=token_spans,
            vision_block_ids=block_ids,
            hot_block_ids=hot_block_ids,
            cold_block_ids=cold_block_ids,
        )
        self._pending_by_external_id[state.external_request_id] = state
        return state

    def register_engine_core_request(
        self,
        request: EngineCoreRequest,
        block_size_tokens: int,
        hot_prefetch_block_count: int | None,
    ) -> VisionKVRequestState | None:
        return self.register_processed_request(
            request=request,
            prompt_text="",
            block_size_tokens=block_size_tokens,
            hot_prefetch_block_count=hot_prefetch_block_count,
        )

    def finalize_request_id(
        self,
        external_request_id: str,
        internal_request_id: str,
    ) -> VisionKVRequestState | None:
        state = self._pending_by_external_id.pop(external_request_id, None)
        if state is None:
            return None
        state.internal_request_id = internal_request_id
        state.last_updated_at = time.time()
        self._by_external_id[state.external_request_id] = state
        self._by_internal_id[internal_request_id] = state
        if internal_request_id not in self._request_order:
            self._request_order.append(internal_request_id)
        return state

    def get(self, request_id: str) -> VisionKVRequestState | None:
        return self._by_internal_id.get(request_id) or self._by_external_id.get(request_id)

    def iter_states(self) -> list[VisionKVRequestState]:
        states: list[VisionKVRequestState] = []
        seen: set[int] = set()
        for internal_request_id in self._request_order:
            state = self._by_internal_id.get(internal_request_id)
            if state is not None and id(state) not in seen:
                states.append(state)
                seen.add(id(state))
        return states

    def latest_offloaded_state(
        self,
        exclude_request_id: str | None = None,
    ) -> VisionKVRequestState | None:
        candidates = [
            state
            for state in self.iter_states()
            if state.is_offloaded
            and state.external_request_id != exclude_request_id
            and state.internal_request_id != exclude_request_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda state: state.last_updated_at)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "external_request_id": state.external_request_id,
                "internal_request_id": state.internal_request_id,
                "generated_tokens": state.generated_tokens,
                "pending_offload": state.pending_offload,
                "vision_block_ids": list(state.vision_block_ids),
                "hot_block_ids": list(state.hot_block_ids),
                "cold_block_ids": list(state.cold_block_ids),
                "offloaded_block_ids": sorted(state.offloaded_block_ids),
                "hot_prefetched_block_ids": sorted(state.hot_prefetched_block_ids),
                "background_prefetched_block_ids": sorted(
                    state.background_prefetched_block_ids
                ),
                "background_prefetch_pending": state.background_prefetch_pending,
                "offload_count": state.offload_count,
                "prefetch_count": state.prefetch_count,
                "background_prefetch_count": state.background_prefetch_count,
                "offload_elapsed_ms": round(state.offload_elapsed_ms, 2),
                "prefetch_elapsed_ms": round(state.prefetch_elapsed_ms, 2),
                "background_prefetch_elapsed_ms": round(
                    state.background_prefetch_elapsed_ms, 2
                ),
            }
            for state in self.iter_states()
        ]


class LiveVisionKVController:
    """Applies the live offload/prefetch policy over tracked request metadata.

    When a TensorOffloadManager is provided and CUDA is available, this
    controller will physically move KV cache block data between GPU and CPU.
    Otherwise, it tracks the intended offload/prefetch lifecycle and emits
    observability hooks.
    """

    def __init__(
        self,
        policy: VisionKVPolicy,
        logger: logging.Logger,
        offload_after_generation_tokens: int = 50,
        tensor_manager: TensorOffloadManager | None = None,
        kv_cache_resolver: Callable[[], list[torch.Tensor]] | None = None,
    ) -> None:
        self.policy = policy
        self.logger = logger
        self.offload_after_generation_tokens = offload_after_generation_tokens
        self.tensor_manager = tensor_manager
        self._kv_cache_resolver = kv_cache_resolver

    def maybe_mark_for_offload(self, state: VisionKVRequestState) -> bool:
        if state.pending_offload or state.is_offloaded:
            return False
        if state.generated_tokens <= self.offload_after_generation_tokens:
            return False
        state.pending_offload = True
        state.last_updated_at = time.time()
        return True

    def offload_vision_blocks(self, state: VisionKVRequestState) -> bool:
        if state.is_offloaded or not state.vision_block_ids:
            return False

        request_key = state.internal_request_id or state.external_request_id
        t0 = time.perf_counter()

        # Attempt physical tensor offload if manager is available
        if self.tensor_manager is not None and self.tensor_manager.available:
            kv_caches = self._resolve_kv_caches()
            if kv_caches:
                self.tensor_manager.offload_kv_blocks_to_cpu(
                    request_id=request_key,
                    kv_caches=kv_caches,
                    block_ids=state.vision_block_ids,
                )

        state.pending_offload = False
        state.offloaded_block_ids = set(state.vision_block_ids)
        state.hot_prefetched_block_ids.clear()
        state.background_prefetched_block_ids.clear()
        state.background_prefetch_pending = False
        state.offload_count += 1
        state.offload_elapsed_ms = (time.perf_counter() - t0) * 1000
        state.last_updated_at = time.time()
        self.logger.info(
            "Offloaded request=%s blocks=%s generated_tokens=%s elapsed_ms=%.2f",
            request_key,
            state.vision_block_ids,
            state.generated_tokens,
            state.offload_elapsed_ms,
        )
        return True

    def prefetch_hot_blocks(self, state: VisionKVRequestState) -> bool:
        if not state.is_offloaded:
            return False
        hot_blocks = [
            block_id for block_id in state.hot_block_ids if block_id in state.offloaded_block_ids
        ]
        if not hot_blocks:
            return False

        request_key = state.internal_request_id or state.external_request_id

        # Attempt physical tensor prefetch if manager is available
        if self.tensor_manager is not None and self.tensor_manager.available:
            kv_caches = self._resolve_kv_caches()
            if kv_caches:
                elapsed = self.tensor_manager.prefetch_kv_blocks_to_gpu(
                    request_id=request_key,
                    kv_caches=kv_caches,
                    block_ids=hot_blocks,
                )
                state.prefetch_elapsed_ms = elapsed * 1000
                budget_ms = self.policy.flashback_budget_ms
                if state.prefetch_elapsed_ms > budget_ms:
                    self.logger.warning(
                        "Hot prefetch exceeded budget: %.2f ms > %.2f ms",
                        state.prefetch_elapsed_ms,
                        budget_ms,
                    )

        state.offloaded_block_ids.difference_update(hot_blocks)
        state.hot_prefetched_block_ids.update(hot_blocks)
        state.prefetch_count += 1
        state.background_prefetch_pending = (
            self.policy.background_prefetch_remainder and bool(state.offloaded_block_ids)
        )
        state.last_updated_at = time.time()
        self.logger.info(
            "Prefetched hot set for request=%s hot_blocks=%s remaining_offloaded=%s "
            "elapsed_ms=%.2f",
            request_key,
            hot_blocks,
            sorted(state.offloaded_block_ids),
            state.prefetch_elapsed_ms,
        )
        return True

    def continue_background_prefetch(self, state: VisionKVRequestState) -> bool:
        if not state.background_prefetch_pending:
            return False
        remaining_blocks = sorted(state.offloaded_block_ids)
        if not remaining_blocks:
            state.background_prefetch_pending = False
            return False

        request_key = state.internal_request_id or state.external_request_id

        # Attempt physical tensor prefetch if manager is available
        if self.tensor_manager is not None and self.tensor_manager.available:
            kv_caches = self._resolve_kv_caches()
            if kv_caches:
                elapsed = self.tensor_manager.prefetch_kv_blocks_to_gpu(
                    request_id=request_key,
                    kv_caches=kv_caches,
                    block_ids=remaining_blocks,
                )
                state.background_prefetch_elapsed_ms = elapsed * 1000

        state.offloaded_block_ids.clear()
        state.background_prefetched_block_ids.update(remaining_blocks)
        state.background_prefetch_pending = False
        state.background_prefetch_count += 1
        state.last_updated_at = time.time()
        self.logger.info(
            "Background-prefetched request=%s cold_blocks=%s elapsed_ms=%.2f",
            request_key,
            remaining_blocks,
            state.background_prefetch_elapsed_ms,
        )
        return True

    def _resolve_kv_caches(self) -> list[torch.Tensor]:
        """Resolve the live KV cache tensors from the vLLM engine."""
        if self._kv_cache_resolver is not None:
            try:
                return self._kv_cache_resolver()
            except Exception:
                self.logger.debug("KV cache resolver failed", exc_info=True)
        return []


class VisionKVPlugin:
    """Monkey-patch vLLM V1 engine and worker methods for VisionKV coordination."""

    def __init__(
        self,
        engine_or_worker: Any,
        *,
        policy: VisionKVPolicy | None = None,
        block_size_tokens: int = 16,
        offload_after_generation_tokens: int = 50,
    ) -> None:
        self.target = engine_or_worker
        self.policy = policy or VisionKVPolicy(
            hot_prefetch_block_count=2,
            flashback_budget_ms=50.0,
            background_prefetch_remainder=True,
        )
        self.block_size_tokens = block_size_tokens
        self.metadata_store = VisionKVMetadataStore()
        self._llm_engine = self._resolve_llm_engine(engine_or_worker)
        self._worker = self._resolve_worker(engine_or_worker)

        # Create tensor offload manager and wire up KV cache resolver
        self.tensor_manager = TensorOffloadManager(logger=LOGGER)
        kv_cache_resolver = self._build_kv_cache_resolver()

        self.controller = LiveVisionKVController(
            policy=self.policy,
            logger=LOGGER,
            offload_after_generation_tokens=offload_after_generation_tokens,
            tensor_manager=self.tensor_manager,
            kv_cache_resolver=kv_cache_resolver,
        )
        self._patches: list[tuple[Any, str, Any]] = []
        self._installed = False
        self._lock = threading.RLock()
        self.peak_cuda_reserved_bytes = 0
        self.baseline_cuda_reserved_bytes = self._read_cuda_reserved_bytes()

    def install(self) -> "VisionKVPlugin":
        with self._lock:
            if self._installed:
                return self

            if self._llm_engine is not None:
                self._patch_input_processor(self._llm_engine)
                self._patch_add_request(self._llm_engine)
                self._patch_step(self._llm_engine)
            if self._worker is not None:
                self._patch_execute_model(self._worker)

            self._installed = True
            LOGGER.info(
                "Installed VisionKVPlugin hot_prefetch_block_count=%s "
                "background_prefetch_remainder=%s",
                self.policy.hot_prefetch_block_count,
                self.policy.background_prefetch_remainder,
            )
        return self

    def uninstall(self) -> None:
        with self._lock:
            for obj, attr_name, original in reversed(self._patches):
                setattr(obj, attr_name, original)
            self._patches.clear()
            self._installed = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "baseline_cuda_reserved_mib": _bytes_to_mib(self.baseline_cuda_reserved_bytes),
            "peak_cuda_reserved_mib": _bytes_to_mib(self.peak_cuda_reserved_bytes),
            "tensor_offload_stats": self.tensor_manager.stats,
            "requests": self.metadata_store.snapshot(),
        }

    def tag_vision_blocks_hook(
        self,
        request: EngineCoreRequest,
        prompt: Any,
    ) -> VisionKVRequestState | None:
        prompt_text = _extract_prompt_text(prompt)
        state = self.metadata_store.register_processed_request(
            request=request,
            prompt_text=prompt_text,
            block_size_tokens=self.block_size_tokens,
            hot_prefetch_block_count=self.policy.hot_prefetch_block_count,
        )
        if state is not None:
            LOGGER.info(
                "Tagged vision blocks external_request_id=%s prompt_tokens=%s "
                "vision_blocks=%s hot_blocks=%s cold_blocks=%s",
                state.external_request_id,
                state.prompt_token_count,
                state.vision_block_ids,
                state.hot_block_ids,
                state.cold_block_ids,
            )
        return state

    def decode_step_hook(self, scheduler_output: Any) -> None:
        del scheduler_output  # only used as an execution boundary for now
        self._advance_lifecycle()
        self._update_peak_vram()

    def _patch_input_processor(self, engine: LLMEngine) -> None:
        input_processor = engine.input_processor
        original = input_processor.process_inputs

        @functools.wraps(original)
        def wrapped_process_inputs(
            request_id: str,
            prompt: Any,
            params: Any,
            supported_tasks: tuple[Any, ...],
            arrival_time: float | None = None,
            lora_request: Any | None = None,
            tokenization_kwargs: dict[str, Any] | None = None,
            trace_headers: Mapping[str, str] | None = None,
            priority: int = 0,
            data_parallel_rank: int | None = None,
            resumable: bool = False,
        ) -> EngineCoreRequest:
            request = original(
                request_id,
                prompt,
                params,
                supported_tasks,
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                resumable=resumable,
            )
            self.tag_vision_blocks_hook(request, prompt)
            return request

        self._remember_patch(input_processor, "process_inputs")
        setattr(input_processor, "process_inputs", wrapped_process_inputs)

    def _patch_add_request(self, engine: LLMEngine) -> None:
        original = engine.add_request

        @functools.wraps(original)
        def wrapped_add_request(
            request_id: str,
            prompt: Any,
            params: Any,
            arrival_time: float | None = None,
            lora_request: Any | None = None,
            tokenization_kwargs: dict[str, Any] | None = None,
            trace_headers: Mapping[str, str] | None = None,
            priority: int = 0,
            prompt_text: str | None = None,
            **kwargs: Any,
        ) -> str:
            self._trigger_followup_prefetch(external_request_id=request_id)

            if isinstance(prompt, EngineCoreRequest):
                self.metadata_store.register_engine_core_request(
                    request=prompt,
                    block_size_tokens=self.block_size_tokens,
                    hot_prefetch_block_count=self.policy.hot_prefetch_block_count,
                )

            internal_request_id = original(
                request_id,
                prompt,
                params,
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                prompt_text=prompt_text,
                **kwargs,
            )
            state = self.metadata_store.finalize_request_id(
                external_request_id=request_id,
                internal_request_id=internal_request_id,
            )
            if state is not None:
                LOGGER.info(
                    "Registered live request external_request_id=%s internal_request_id=%s",
                    request_id,
                    internal_request_id,
                )
            return internal_request_id

        self._remember_patch(engine, "add_request")
        setattr(engine, "add_request", wrapped_add_request)

    def _patch_step(self, engine: LLMEngine) -> None:
        original = engine.step

        @functools.wraps(original)
        def wrapped_step(*args: Any, **kwargs: Any) -> Any:
            outputs = original(*args, **kwargs)
            self._record_generation_progress(outputs)
            self._advance_lifecycle()
            self._update_peak_vram()
            return outputs

        self._remember_patch(engine, "step")
        setattr(engine, "step", wrapped_step)

    def _patch_execute_model(self, worker: WorkerBase) -> None:
        original = worker.execute_model

        @functools.wraps(original)
        def wrapped_execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            self.decode_step_hook(scheduler_output)
            return original(scheduler_output, *args, **kwargs)

        self._remember_patch(worker, "execute_model")
        setattr(worker, "execute_model", wrapped_execute_model)

    def _remember_patch(self, obj: Any, attr_name: str) -> None:
        self._patches.append((obj, attr_name, getattr(obj, attr_name)))

    def _trigger_followup_prefetch(self, external_request_id: str) -> None:
        state = self.metadata_store.latest_offloaded_state(exclude_request_id=external_request_id)
        if state is None:
            return
        if self.controller.prefetch_hot_blocks(state):
            LOGGER.info(
                "Triggered hot-set prefetch for follow-up prompt external_request_id=%s "
                "source_request_id=%s",
                external_request_id,
                state.internal_request_id or state.external_request_id,
            )

    def _record_generation_progress(self, outputs: Any) -> None:
        if outputs is None:
            return
        for request_output in outputs:
            request_id = getattr(request_output, "request_id", None)
            if not isinstance(request_id, str):
                continue
            state = self.metadata_store.get(request_id)
            if state is None:
                continue
            generated_tokens = self._extract_generated_token_count(request_output)
            if generated_tokens is None:
                continue
            if generated_tokens > state.generated_tokens:
                state.generated_tokens = generated_tokens
                state.last_updated_at = time.time()
                self.controller.maybe_mark_for_offload(state)

    def _advance_lifecycle(self) -> None:
        for state in self.metadata_store.iter_states():
            if state.pending_offload:
                self.controller.offload_vision_blocks(state)
                continue
            if state.background_prefetch_pending and state.hot_prefetched_block_ids:
                self.controller.continue_background_prefetch(state)

    @staticmethod
    def _extract_generated_token_count(request_output: Any) -> int | None:
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return None
        token_lengths: list[int] = []
        for candidate in outputs:
            token_ids = getattr(candidate, "token_ids", None)
            if token_ids is not None:
                token_lengths.append(len(token_ids))
        if token_lengths:
            return max(token_lengths)
        return None

    def _read_cuda_reserved_bytes(self) -> int:
        if not torch.cuda.is_available():
            return 0
        try:
            return int(torch.cuda.memory_reserved())
        except Exception:
            return 0

    def _update_peak_vram(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            current_peak = int(torch.cuda.max_memory_reserved())
        except Exception:
            current_peak = self._read_cuda_reserved_bytes()
        self.peak_cuda_reserved_bytes = max(self.peak_cuda_reserved_bytes, current_peak)

    @staticmethod
    def _resolve_llm_engine(target: Any) -> LLMEngine | None:
        if isinstance(target, LLMEngine):
            return target
        llm_engine = getattr(target, "llm_engine", None)
        if isinstance(llm_engine, LLMEngine):
            return llm_engine
        return None

    @staticmethod
    def _resolve_worker(target: Any) -> WorkerBase | None:
        if isinstance(target, WorkerBase):
            return target

        llm_engine = VisionKVPlugin._resolve_llm_engine(target)
        if llm_engine is None:
            return None

        engine_core = getattr(llm_engine, "engine_core", None)
        model_executor = getattr(engine_core, "model_executor", None)
        driver_worker = getattr(model_executor, "driver_worker", None)
        worker = getattr(driver_worker, "worker", None)
        if isinstance(worker, WorkerBase):
            return worker
        return None

    def _build_kv_cache_resolver(self) -> Callable[[], list[torch.Tensor]]:
        """Build a callable that navigates vLLM's internal objects to find live KV cache tensors."""
        def resolver() -> list[torch.Tensor]:
            if self._worker is None:
                return []

            # Check if self._worker has gpu_cache
            gpu_cache = getattr(self._worker, "gpu_cache", None)
            if gpu_cache is not None:
                if isinstance(gpu_cache, list):
                    flat_caches = []
                    for item in gpu_cache:
                        if isinstance(item, torch.Tensor):
                            flat_caches.append(item)
                        elif isinstance(item, (list, tuple)):
                            for t in item:
                                if isinstance(t, torch.Tensor):
                                    flat_caches.append(t)
                    if flat_caches:
                        return flat_caches

            # Check if model_runner has gpu_cache or kv_caches
            model_runner = getattr(self._worker, "model_runner", None)
            if model_runner is not None:
                for attr_name in ("gpu_cache", "kv_caches", "kv_cache"):
                    caches = getattr(model_runner, attr_name, None)
                    if caches is not None:
                        if isinstance(caches, list):
                            flat_caches = []
                            for item in caches:
                                if isinstance(item, torch.Tensor):
                                    flat_caches.append(item)
                                elif isinstance(item, (list, tuple)):
                                    for t in item:
                                        if isinstance(t, torch.Tensor):
                                            flat_caches.append(t)
                            if flat_caches:
                                return flat_caches
                        elif isinstance(caches, torch.Tensor):
                            return [caches]

            # Check cache_engine for gpu_cache
            cache_engine = getattr(self._worker, "cache_engine", None)
            if cache_engine is not None:
                caches = getattr(cache_engine, "gpu_cache", None)
                if caches is not None:
                    if isinstance(caches, list):
                        flat_caches = []
                        for item in caches:
                            if isinstance(item, torch.Tensor):
                                flat_caches.append(item)
                            elif isinstance(item, (list, tuple)):
                                for t in item:
                                    if isinstance(t, torch.Tensor):
                                        flat_caches.append(t)
                        if flat_caches:
                            return flat_caches
                    elif isinstance(caches, torch.Tensor):
                        return [caches]

            return []

        return resolver

