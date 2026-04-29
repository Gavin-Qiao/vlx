# vserve 0.5.2b2 Release Notes

`v0.5.2b2` is a beta remediation and hardening release for the vLLM 0.20 runtime transition. It includes the full backend-runtime overhaul work that has landed on `main` since `v0.5.2b1`, plus the final release-audit fixes needed before publishing a new beta.

## Upgrade

Beta/pre-release installs remain the primary channel while vserve is in beta:

```bash
uv tool install --prerelease allow vserve
pip install --pre vserve
vserve update --nightly
```

For existing installs, upgrade to this beta with:

```bash
uv tool upgrade vserve --prerelease allow
```

or:

```bash
pip install --pre --upgrade vserve
```

## Release-Audit Fixes Since 0.5.2b1

- Bumped package metadata from `0.5.2b1` to `0.5.2b2` across `pyproject.toml`, `src/vserve/__init__.py`, `uv.lock`, and README so the release will not collide with the already-published `0.5.2b1`.
- Made generated vLLM YAML profiles service-readable. `write_profile_yaml()` now writes profiles with mode `0644`, preventing systemd service users such as `vllm` from being unable to read launch configs created by an operator account.
- Hardened `vserve cache clean --all` against symlinked parent components inside the vLLM root. The command now refuses unsafe cache paths before it runs `du`, `find`, or `rm`, including cases where `root/.cache` is a symlink to an external directory.
- Made the fan daemon honor the configured GPU index instead of hardcoding GPU `0`. Temperature reads, daemon startup, and fixed-speed mode now use `cfg().gpu_index`.
- Removed stale local build artifacts from `dist/` and rebuilt only `vserve-0.5.2b2` wheel and sdist.
- Updated README test-count references to the verified `495` test suite.

## Backend Runtime Overhaul

- Replaced thin backend handling with typed backend contracts covering runtime information, compatibility checks, config building, capabilities, tuning results, active manifests, profiles, service status, and cache fingerprints.
- Moved backend-specific runtime/config/status behavior out of the CLI and into backend-aware paths.
- Added backend-aware runtime compatibility checks, including a dedicated `vserve.runtime` module for collecting vLLM runtime identity and validating the supported range.
- Pinned supported stable vLLM runtime compatibility to `>=0.20,<0.21`.
- Added backend runtime fingerprints so cached limits can be invalidated when runtime, model, GPU policy, or parser-relevant metadata changes.

## Scripted Serving and Profiles

- Expanded `vserve run` into a scriptable command that accepts model query terms plus non-interactive flags:
  - `--profile`
  - `--save-profile`
  - `--yes`
  - `--replace`
  - `--context`
  - `--slots`
  - `--kv-cache-dtype`
  - `--batched-tokens`
  - `--gpu-util`
  - `--port`
  - `--tools`
  - `--tool-parser`
  - `--reasoning-parser`
  - `--trust-remote-code`
  - llama.cpp-specific `--gpu-layers`, `--embedding`, and `--pooling`
- `run --yes` is now intentionally non-interactive. It refuses to replace a running backend unless `--replace` is provided.
- `run --yes` now derives omitted defaults from tuned limits and can auto-tune when required defaults are missing or stale.
- `run --profile` supports profile names and explicit profile paths for launch.
- Added `vserve profile list`, `vserve profile show`, and `vserve profile rm`.
- Restricted profile management commands to known vserve profile roots. External profile paths remain launch-only through `run --profile`.
- Restricted `--save-profile` to safe profile names containing only letters, numbers, `.`, `_`, and `-`, excluding `.`/`..` and path separators.

## Security and Filesystem Safety

- Defaulted vLLM `trust-remote-code` to false.
- Added explicit `vserve run --trust-remote-code`; generated vLLM configs include `trust-remote-code` only when the operator opts in.
- Hardened lock/session state handling with no-follow file operations and regular-file checks.
- Fixed `VserveLock.release()` to preserve stable lock files by truncating metadata while locked instead of unlinking the lock file.
- Added symlink refusal for lock/session/fan state paths.
- Hardened fan state writes against symlink clobbering.
- Added root and cache safety checks before destructive cache operations.
- Added service-name validation before privileged `sudo systemctl start/stop`.
- Added best-effort verification that systemd units belong to the intended vserve backend before privileged systemctl actions.

## Operator and Lifecycle Safety

- Split read-only `systemctl is-active` from privileged `sudo systemctl start/stop`.
- Added non-interactive `sudo -n` behavior for automated `run --yes` start/stop paths.
- Locked `runtime upgrade` and cache-clean operations against active or uncertain backend state.
- `cache clean` now acquires the GPU lifecycle lock, re-probes backend state under that lock, supports `--dry-run`, previews sizes where possible, refuses active/uncertain backend state, and exits nonzero on failed delete operations.
- `stop` is more defensive when backend state probes are uncertain and can issue fallback stop requests to known backends.
- Launch health behavior is stricter for automation. Non-interactive launches exit nonzero if the API health endpoint does not become healthy, even if the service remains active and warming.
- Launch manifests now record starting, warming, ready, and failed states more truthfully.
- llama.cpp startup now restores previous active links and writes a failed active manifest on start failure.

## Status and Doctor Truthfulness

- Added `vserve status --json`.
- Status now collects per-backend config, manifest, probe error, port, endpoint, and health-related information.
- Status now reports multiple active backends and gives next-action guidance instead of claiming readiness when the manifest is starting, warming, failed, or uncertain.
- Added `vserve doctor --json` and `vserve doctor --strict`.
- Doctor now treats vLLM checks as required only when vLLM is configured or no llama.cpp-only setup is present.
- Doctor now fails strict mode when the vLLM systemd unit does not load the exact generated `configs/.env` required for `CUDA_VISIBLE_DEVICES`.
- Doctor no longer claims a backend is serving merely because a port is open; it checks the active backend's configured port and health where possible.
- Machine-readable `status --json` and `doctor --json` suppress background update-check side effects.

## vLLM 0.20 Compatibility

- Parser registry introspection now runs under the configured vLLM runtime Python instead of the CLI process.
- Tool and reasoning parser validation supports current vLLM parser registry APIs, including `list_registered()`.
- Tool parser and reasoning parser validation are independent; reasoning parser configuration no longer depends on tool calling.
- `run --yes --tools` resolves parser support through cached limits only when the parser is supported by the configured runtime, then falls back to template detection, then fails clearly.
- vLLM generated startup state records `gpu_index` and `CUDA_VISIBLE_DEVICES`.
- vLLM startup writes `configs/.env` with `CUDA_VISIBLE_DEVICES=<gpu.index>` and refuses to pretend GPU selection is enforced if the systemd unit does not consume that file.

## llama.cpp

- llama.cpp configs/scripts now carry `gpu_index` and generated `active.sh` exports `CUDA_VISIBLE_DEVICES=<index>`.
- llama.cpp runtime info calls `llama-server --version` and compatibility fails clearly if no entrypoint exists.
- llama.cpp cached limits now include versioned fingerprint checks with runtime identity and GPU policy.
- llama.cpp interactive and scripted config paths retune when cached limits are missing or stale.
- Scripted llama.cpp defaults reject missing tuned `n_gpu_layers` instead of silently falling back to an unsafe value.
- Embedding and pooling flags are available through scripted run config.

## Model and Variant Correctness

- Model matching commands now accept `MODEL...` query terms and use tokenized/ranked fuzzy matching across provider, model name, quant, backend, and format.
- vLLM serving now requires top-level runnable safetensors or `.bin` weights.
- `.bin` weights are included in model size calculations.
- MHA configs without `num_key_value_heads` now fall back to `num_attention_heads` during tuning.
- Split GGUF shards are grouped as one runnable variant only when complete.
- Incomplete safetensors/bin indexed variants are skipped rather than leaked into shared files.
- Local model detection rejects indexed model roots with missing referenced shards.
- Mixed GGUF and non-GGUF download selections are rejected with backend-format guidance.
- GGUF downloads now materialize one runnable root per variant, including single-variant downloads, and strip subdirectory prefixes so `.gguf` files land top-level.
- Repeated GGUF downloads clear stale quant files that do not belong to the selected variant root.
- Multiple non-GGUF subdirectory variants materialize into separate runnable roots with shared top-level config/tokenizer files copied into each root.
- Source roots left behind after variant materialization are ignored when they are config-only and not runnable.

## GPU Policy

- Added config support for `gpu.index`, defaulting to `0`.
- GPU memory policy is shared across add, tune, and run.
- Effective GPU memory utilization is bounded to `0.5..0.99`.
- Automatic overhead policy fails with an actionable error when VRAM is invalid or the resolved utilization would be outside the supported range.
- Tuning fingerprints include GPU index and GPU memory policy.
- Multi-GPU `nvidia-smi` parsing is row-aware and validates configured GPU index availability.
- Fan helpers and the fan daemon now honor configured GPU index.

## Update and Release Hygiene

- `vserve update --nightly` supports beta/pre-release upgrade paths for uv and pip.
- pip fallback now prefers the installer that owns the running executable where possible and verifies the upgraded environment instead of blindly claiming success.
- CI and publish workflows run ruff, mypy with `--check-untyped-defs`, and pytest.
- README and troubleshooting docs now cover:
  - beta/pre-release install commands
  - `run --yes` and `--replace`
  - `--trust-remote-code`
  - profile invocation/path rules
  - exact `cache clean --all` scope
  - vLLM runtime upgrade prerequisites
  - GGUF one-root-per-variant behavior
  - `configs/models`, active config links, `.env`, `tmp`, cache dirs, and run manifests
  - GPU index enforcement requirements

## Test Coverage

The suite now includes regression coverage for:

- vLLM profile permissions for service-user readability.
- Symlinked cache-parent refusal before cache clean subprocesses run.
- Fan daemon GPU-index selection.
- Lock/session symlink refusal and stable lock reacquire behavior.
- `run --yes` defaults, auto-tuning, parser resolution, `--replace`, and profile launch behavior.
- `trust-remote-code` default-off behavior and explicit opt-in.
- Profile path/name safety.
- vLLM parser runtime introspection through configured runtime Python.
- llama.cpp and vLLM runtime fingerprint invalidation.
- incomplete safetensors/bin and split-GGUF variant rejection.
- subdirectory and multi-variant model materialization.
- status/doctor JSON behavior, strict exits, multiple active backends, and uncertain probe states.
- cache clean dry-run and failed-delete behavior.
- update fallback behavior and README/troubleshooting command text.

Final local verification for this release:

```bash
uv run ruff check src/ tests/
uv run mypy src/vserve/ --ignore-missing-imports --check-untyped-defs
uv run pytest -q
uv build
```

Observed result:

- ruff passed
- mypy passed
- pytest passed: `495 passed`
- build produced:
  - `dist/vserve-0.5.2b2.tar.gz`
  - `dist/vserve-0.5.2b2-py3-none-any.whl`

## Notes for Operators

- vLLM remote code execution is opt-in. Add `--trust-remote-code` only for model repositories you trust.
- For multi-GPU hosts, set `gpu.index` in `~/.config/vserve/config.yaml` and ensure the vLLM systemd unit loads the generated `configs/.env`.
- For automation, `run --yes` is strict by design. Use `--replace` when the script is intentionally allowed to stop an active backend.
- If cached tuning limits are stale or fingerprint-less, vserve will retune rather than silently reusing them for launch defaults.
- GGUF variants are now intentionally isolated as one runnable model root per quant/shard group.
