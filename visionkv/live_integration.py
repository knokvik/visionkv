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
                "vision_block_ids": list(state.vision_block_ids),
                "hot_block_ids": list(state.hot_block_ids),
                "cold_block_ids": list(state.cold_block_ids),
                "offloaded_block_ids": sorted(state.offloaded_block_ids),
                "hot_prefetched_block_ids": sorted(state.hot_prefetched_block_ids),
                "background_prefetched_block_ids": sorted(
                    state.background_prefetched_block_ids
                ),
            }
            for state in self.iter_states()
        ]


class LiveVisionKVController:
    """Applies the live offload/prefetch policy over tracked request metadata.

    This controller is intentionally coordination-only. Public pip-installed
    vLLM APIs do not currently expose a safe, request-scoped KV block migration
    surface for external plugins, so the controller tracks the intended
    offload/prefetch lifecycle and emits observability hooks without claiming
    to move private KV tensors directly.
    """

    def __init__(
        self,
        policy: VisionKVPolicy,
        logger: logging.Logger,
        offload_after_generation_tokens: int = 50,
    ) -> None:
        self.policy = policy
        self.logger = logger
        self.offload_after_generation_tokens = offload_after_generation_tokens

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
        state.pending_offload = False
        state.offloaded_block_ids = set(state.vision_block_ids)
        state.hot_prefetched_block_ids.clear()
        state.background_prefetched_block_ids.clear()
        state.background_prefetch_pending = False
        state.offload_count += 1
        state.last_updated_at = time.time()
        self.logger.info(
            "Offloaded request=%s blocks=%s generated_tokens=%s",
            state.internal_request_id or state.external_request_id,
            state.vision_block_ids,
            state.generated_tokens,
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
        state.offloaded_block_ids.difference_update(hot_blocks)
        state.hot_prefetched_block_ids.update(hot_blocks)
        state.prefetch_count += 1
        state.background_prefetch_pending = (
            self.policy.background_prefetch_remainder and bool(state.offloaded_block_ids)
        )
        state.last_updated_at = time.time()
        self.logger.info(
            "Prefetched hot set for request=%s hot_blocks=%s remaining_offloaded=%s",
            state.internal_request_id or state.external_request_id,
            hot_blocks,
            sorted(state.offloaded_block_ids),
        )
        return True

    def continue_background_prefetch(self, state: VisionKVRequestState) -> bool:
        if not state.background_prefetch_pending:
            return False
        remaining_blocks = sorted(state.offloaded_block_ids)
        if not remaining_blocks:
            state.background_prefetch_pending = False
            return False
        state.offloaded_block_ids.clear()
        state.background_prefetched_block_ids.update(remaining_blocks)
        state.background_prefetch_pending = False
        state.background_prefetch_count += 1
        state.last_updated_at = time.time()
        self.logger.info(
            "Background-prefetched request=%s cold_blocks=%s",
            state.internal_request_id or state.external_request_id,
            remaining_blocks,
        )
        return True


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
        self.controller = LiveVisionKVController(
            policy=self.policy,
            logger=LOGGER,
            offload_after_generation_tokens=offload_after_generation_tokens,
        )
        self._patches: list[tuple[Any, str, Any]] = []
        self._installed = False
        self._lock = threading.RLock()
        self._llm_engine = self._resolve_llm_engine(engine_or_worker)
        self._worker = self._resolve_worker(engine_or_worker)
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
            "baseline_cuda_reserved_mib": _bytes_to_mib(self.baseline_cuda_reserved_bytes),
            "peak_cuda_reserved_mib": _bytes_to_mib(self.peak_cuda_reserved_bytes),
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
        self._update_peak_vram()
        for state in self.metadata_store.iter_states():
            if state.pending_offload:
                self.controller.offload_vision_blocks(state)
                continue
            if state.background_prefetch_pending and state.hot_prefetched_block_ids:
                self.controller.continue_background_prefetch(state)

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
