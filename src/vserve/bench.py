"""Small bounded benchmark helpers for local serving endpoints."""

from __future__ import annotations

import json
import math
import time
from typing import Callable, Protocol
from urllib.request import Request, urlopen


Clock = Callable[[], float]


class PostJson(Protocol):
    def __call__(self, url: str, payload: dict, *, timeout: float) -> dict:
        ...


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    data = json.loads(body or "{}")
    return data if isinstance(data, dict) else {}


def _p95_ms(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[idx] * 1000, 2)


def run_openai_completion_benchmark(
    base_url: str,
    *,
    model: str,
    request_count: int = 8,
    timeout_s: float = 60,
    max_tokens: int = 16,
    prompt: str = "Write one concise sentence about GPU inference tuning.",
    post_json: PostJson = _post_json,
    monotonic: Clock = time.monotonic,
) -> dict:
    """Run a sequential, bounded OpenAI-compatible completions benchmark."""
    request_count = max(1, int(request_count))
    timeout_s = max(0.1, float(timeout_s))
    max_tokens = max(1, int(max_tokens))
    started = monotonic()
    deadline = started + timeout_s
    latencies: list[float] = []
    completion_tokens = 0
    errors: list[str] = []
    url = f"{base_url.rstrip('/')}/v1/completions"

    for _ in range(request_count):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        req_started = monotonic()
        try:
            data = post_json(url, payload, timeout=max(0.1, min(remaining, 30.0)))
        except Exception as exc:
            errors.append(str(exc))
            break
        elapsed = max(0.0, monotonic() - req_started)
        latencies.append(elapsed)
        usage = data.get("usage")
        if isinstance(usage, dict):
            value = usage.get("completion_tokens")
            if isinstance(value, int) and not isinstance(value, bool):
                completion_tokens += value

    total_seconds = max(0.0, monotonic() - started)
    completed = len(latencies)
    return {
        "status": "ok" if completed else "error",
        "requests_completed": completed,
        "request_count": request_count,
        "total_seconds": round(total_seconds, 3),
        "mean_latency_ms": round((sum(latencies) / completed) * 1000, 2) if completed else None,
        "p95_latency_ms": _p95_ms(latencies),
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(completion_tokens / total_seconds, 2) if total_seconds > 0 else 0,
        "errors": errors,
    }


def run_openai_embedding_benchmark(
    base_url: str,
    *,
    model: str,
    request_count: int = 8,
    timeout_s: float = 60,
    text: str = "GPU inference benchmark text.",
    post_json: PostJson = _post_json,
    monotonic: Clock = time.monotonic,
) -> dict:
    """Run a sequential, bounded OpenAI-compatible embeddings benchmark."""
    request_count = max(1, int(request_count))
    timeout_s = max(0.1, float(timeout_s))
    started = monotonic()
    deadline = started + timeout_s
    latencies: list[float] = []
    errors: list[str] = []
    dimensions: int | None = None
    url = f"{base_url.rstrip('/')}/v1/embeddings"

    for _ in range(request_count):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        payload = {
            "model": model,
            "input": text,
        }
        req_started = monotonic()
        try:
            data = post_json(url, payload, timeout=max(0.1, min(remaining, 30.0)))
        except Exception as exc:
            errors.append(str(exc))
            break
        elapsed = max(0.0, monotonic() - req_started)
        latencies.append(elapsed)
        values = data.get("data")
        if isinstance(values, list) and values and isinstance(values[0], dict):
            embedding = values[0].get("embedding")
            if isinstance(embedding, list):
                dimensions = len(embedding)

    total_seconds = max(0.0, monotonic() - started)
    completed = len(latencies)
    return {
        "status": "ok" if completed else "error",
        "requests_completed": completed,
        "request_count": request_count,
        "total_seconds": round(total_seconds, 3),
        "mean_latency_ms": round((sum(latencies) / completed) * 1000, 2) if completed else None,
        "p95_latency_ms": _p95_ms(latencies),
        "embedding_dimensions": dimensions,
        "items_per_second": round(completed / total_seconds, 2) if total_seconds > 0 else 0,
        "errors": errors,
    }
