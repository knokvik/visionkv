"""Crash-test harness for a real VLM served by vLLM.

This harness targets a vLLM server exposing the OpenAI-compatible API.
It does not prove KV-cache reuse by itself, but it gives us a repeatable way to:
- send multimodal prompts with a real image,
- perform a two-turn scenario with a follow-up image question,
- capture latency and GPU memory snapshots,
- detect obvious failures such as OOMs, empty replies, or HTTP errors.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class VisionConversationScenario:
    model: str
    image_reference: str
    initial_prompt: str
    followup_prompt: str
    max_tokens: int = 500


@dataclass
class TurnResult:
    prompt: str
    latency_ms: float
    response_text: str
    gpu_memory_before_mb: Optional[int]
    gpu_memory_after_mb: Optional[int]


@dataclass
class IntegrationOutcome:
    status: str
    turns: List[TurnResult]
    error: Optional[str] = None


def sample_gpu_memory_mb() -> Optional[int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first_line = completed.stdout.strip().splitlines()[0]
    return int(first_line)


def _to_image_url(image_reference: str) -> str:
    if image_reference.startswith(("http://", "https://", "data:")):
        return image_reference

    image_path = Path(image_reference)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_message(prompt: str, image_reference: str) -> Dict[str, object]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _to_image_url(image_reference)}},
        ],
    }


class OpenAIServerHarness:
    """Runs the two-turn crash-test scenario against a vLLM server."""

    def __init__(self, base_url: str, api_key: str = "EMPTY") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def run_two_turn_scenario(self, scenario: VisionConversationScenario) -> IntegrationOutcome:
        turns: List[TurnResult] = []

        try:
            first_turn = self._run_turn(
                scenario.model,
                messages=[build_multimodal_message(scenario.initial_prompt, scenario.image_reference)],
                max_tokens=scenario.max_tokens,
            )
            turns.append(first_turn)

            second_turn = self._run_turn(
                scenario.model,
                messages=[
                    build_multimodal_message(scenario.initial_prompt, scenario.image_reference),
                    {"role": "assistant", "content": first_turn.response_text},
                    {"role": "user", "content": scenario.followup_prompt},
                ],
                max_tokens=scenario.max_tokens,
            )
            turns.append(second_turn)
        except RuntimeError as exc:
            return IntegrationOutcome(status="failed", turns=turns, error=str(exc))

        if not all(turn.response_text.strip() for turn in turns):
            return IntegrationOutcome(
                status="failed",
                turns=turns,
                error="Model returned an empty response.",
            )

        return IntegrationOutcome(status="passed", turns=turns)

    def _run_turn(
        self,
        model: str,
        messages: List[Dict[str, object]],
        max_tokens: int,
    ) -> TurnResult:
        gpu_before = sample_gpu_memory_mb()
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Request failed: {exc.reason}") from exc

        latency_ms = (time.perf_counter() - start) * 1000
        gpu_after = sample_gpu_memory_mb()
        response_json = json.loads(body)
        response_text = response_json["choices"][0]["message"]["content"]

        return TurnResult(
            prompt=str(messages[-1]["content"]),
            latency_ms=latency_ms,
            response_text=response_text,
            gpu_memory_before_mb=gpu_before,
            gpu_memory_after_mb=gpu_after,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a VisionKV crash-test scenario.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--initial-prompt", required=True)
    parser.add_argument("--followup-prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    harness = OpenAIServerHarness(base_url=args.base_url, api_key=args.api_key)
    scenario = VisionConversationScenario(
        model=args.model,
        image_reference=args.image_reference,
        initial_prompt=args.initial_prompt,
        followup_prompt=args.followup_prompt,
        max_tokens=args.max_tokens,
    )
    outcome = harness.run_two_turn_scenario(scenario)
    print(json.dumps(asdict(outcome), indent=2))
    return 0 if outcome.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
