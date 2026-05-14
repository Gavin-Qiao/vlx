# vserve 0.5.6 Release Notes

`v0.5.6` is the SOTA tuning and release-hygiene release for vserve's vLLM and llama.cpp backends.

## Upgrade

```bash
uv tool upgrade vserve
```

or:

```bash
pip install --upgrade vserve
```

For optional GGUF metadata support from the upstream `gguf` package:

```bash
pip install --upgrade 'vserve[llamacpp]'
```

## vLLM Tuning Refurbish

- Reworked vLLM capacity math around vLLM V1 serving assumptions instead of a single native-KV estimate.
- KV cache capacity now rounds resident tokens to PagedAttention block boundaries.
- Tuning reports native, FP8, and TurboQuant KV-cache dtype profiles with per-token bytes and compression ratios.
- TurboQuant profiles include `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, and `turboquant_3bit_nc`.
- Recommended profiles now separate interactivity, balanced, and throughput use cases with chunked-prefill-oriented batched-token budgets.
- Generated vLLM configs can carry `performance-mode`, `optimization-level`, `block-size`, `kv-cache-memory-bytes`, and prefix-cache choices.
- Runtime compatibility remains pinned to stable vLLM `>=0.20,<0.21`.

## llama.cpp GGUF Tuning

- Added a direct GGUF metadata reader so tuning can use model metadata even when the optional `gguf` Python package is unavailable.
- llama.cpp tuning now uses GGUF-declared context length instead of falling back to conservative 4k estimates.
- KV capacity math handles layerwise KV heads, key/value dimensions, sliding-window attention cache cells, and recurrent state bytes.
- Qwen hybrid/recurrent GGUF models now count only full-attention layers against long-context KV capacity.
- Gemma sliding-window GGUF models now cap SWA cache growth by the window plus batch reserve.
- Split GGUF variants are still grouped and validated as coherent runnable models.

## Benchmarking

- Added `vserve tune --bench` for opt-in bounded microbenchmarks of tuned vLLM and llama.cpp profiles.
- Completion and embedding benchmark helpers use OpenAI-compatible endpoints and hard request/time bounds.
- Benchmark startup timeout, candidate count, request count, and measurement timeout are recorded in saved limits.
- Downloads no longer run benchmarks implicitly. `vserve add` analytically tunes and saves limits without starting a serving backend.

## UX And Reporting

- Model listing now uses "Weights" size terminology, which is clearer for both vLLM and GGUF models.
- `vserve tune` summaries show benchmark status when explicit benchmark results are available.
- README install, command, tuning, and test-count references now match the current release.

## Packaging And Release Hygiene

- Bumped package metadata to `0.5.6` to avoid the already-published `0.5.5` release and the yanked `0.6.0`.
- Added sdist excludes for local tool/cache directories such as `.claude`, `.codex`, `.serena`, `.worktrees`, and generated docs scratch space.
- Ensured the new benchmark helper module and tests are part of the release commit.

## Verification

Fresh local release checks before tagging:

- `uv run ruff check src/ tests/`
- `uv run mypy src/vserve/ --ignore-missing-imports --check-untyped-defs`
- `uv run pytest tests/ -q --tb=long`
- `uv build --out-dir /tmp/vserve-build-audit`
- sdist/wheel artifact inspection for local-config leakage and required package files
