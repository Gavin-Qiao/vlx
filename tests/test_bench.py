"""Tests for bounded benchmark helpers."""


def test_openai_completion_benchmark_counts_requests_and_tokens():
    from vserve.bench import run_openai_completion_benchmark

    calls: list[dict] = []

    def post_json(_url: str, payload: dict, *, timeout: float) -> dict:
        calls.append(payload)
        assert timeout <= 10
        return {"usage": {"completion_tokens": 7}}

    result = run_openai_completion_benchmark(
        "http://localhost:8888",
        model="/models/test",
        request_count=3,
        timeout_s=10,
        max_tokens=8,
        post_json=post_json,
    )

    assert len(calls) == 3
    assert result["status"] == "ok"
    assert result["requests_completed"] == 3
    assert result["completion_tokens"] == 21
    assert result["tokens_per_second"] >= 0


def test_openai_completion_benchmark_stops_at_time_budget():
    from vserve.bench import run_openai_completion_benchmark

    calls = 0
    values = [0.0, 0.0, 0.0, 0.6, 0.6, 0.6]

    def monotonic() -> float:
        return values.pop(0) if values else 0.6

    def post_json(_url: str, _payload: dict, *, timeout: float) -> dict:
        nonlocal calls
        calls += 1
        return {"usage": {"completion_tokens": 1}}

    result = run_openai_completion_benchmark(
        "http://localhost:8888",
        model="/models/test",
        request_count=10,
        timeout_s=0.5,
        post_json=post_json,
        monotonic=monotonic,
    )

    assert calls == 1
    assert result["status"] == "ok"
    assert result["requests_completed"] == 1


def test_openai_embedding_benchmark_uses_embeddings_endpoint():
    from vserve.bench import run_openai_embedding_benchmark

    calls: list[tuple[str, dict]] = []

    def post_json(url: str, payload: dict, *, timeout: float) -> dict:
        calls.append((url, payload))
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    result = run_openai_embedding_benchmark(
        "http://localhost:8888",
        model="/models/embed",
        request_count=2,
        timeout_s=10,
        post_json=post_json,
    )

    assert len(calls) == 2
    assert calls[0][0] == "http://localhost:8888/v1/embeddings"
    assert calls[0][1]["input"]
    assert result["status"] == "ok"
    assert result["requests_completed"] == 2
    assert result["embedding_dimensions"] == 3
