"""Benchmark orchestration — param generation and winner selection."""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROFILES = {
    "interactive": {
        "subcommand": "serve",
        "metric": "mean_ttft_ms",
        "direction": "minimize",
    },
    "batch": {
        "subcommand": "serve_workload",
        "metric": "output_throughput",
        "direction": "maximize",
    },
    "agentic": {
        "subcommand": "serve",
        "metric": "mean_ttft_ms",
        "direction": "minimize",
    },
    "creative": {
        "subcommand": "serve",
        "metric": "median_tpot_ms",
        "direction": "minimize",
    },
}


def _max_proven_context(limits: dict, kv_dtype: str | None = None) -> int:
    best = 0
    for ctx_str, dtypes in limits.get("limits", {}).items():
        ctx = int(ctx_str)
        if kv_dtype:
            if dtypes.get(kv_dtype) is not None:
                best = max(best, ctx)
        else:
            if any(v is not None for v in dtypes.values()):
                best = max(best, ctx)
    return best


def _working_kv_dtypes(limits: dict, context_len: int) -> list[str]:
    entry = limits.get("limits", {}).get(str(context_len), {})
    return [k for k, v in entry.items() if v is not None]


def generate_sweep_params(
    *, profile: str, limits: dict, output_dir: Path,
) -> tuple[Path, Path, str]:
    is_moe = limits.get("is_moe", False)
    gpu_mem_util = limits.get("gpu_memory_utilization", 0.958)
    subcommand = PROFILES[profile]["subcommand"]

    serve: dict[str, dict] = {}
    bench: dict[str, dict] = {}

    if profile == "interactive":
        for ctx_str, dtypes in limits.get("limits", {}).items():
            ctx = int(ctx_str)
            for kv, slots in dtypes.items():
                if slots is None:
                    continue
                serve[f"ctx{ctx // 1024}k-kv_{kv}"] = {
                    "max_model_len": ctx,
                    "kv_cache_dtype": kv,
                    "enable_prefix_caching": True,
                    "gpu_memory_utilization": gpu_mem_util,
                }
        for il in [256, 512, 2048]:
            for ol in [128, 256]:
                bench[f"in{il}-out{ol}"] = {
                    "random_input_len": il,
                    "random_output_len": ol,
                    "max_concurrency": 1,
                    "num_prompts": 20,
                }

    elif profile == "batch":
        max_seqs_values = [8, 16, 32, 64]
        if is_moe:
            max_seqs_values = [s for s in max_seqs_values if s <= 32]
        for seqs in max_seqs_values:
            for bt in [1024, 2048, 4096]:
                if bt < seqs:
                    continue
                for kv in ("auto", "fp8"):
                    serve[f"seqs{seqs}-bt{bt}-kv_{kv}"] = {
                        "max_num_seqs": seqs,
                        "max_num_batched_tokens": bt,
                        "kv_cache_dtype": kv,
                        "gpu_memory_utilization": gpu_mem_util,
                        "max_model_len": 4096,
                    }
        bench["batch_workload"] = {
            "random_input_len": 512,
            "random_output_len": 256,
            "num_prompts": 50,
        }

    elif profile == "agentic":
        max_ctx = _max_proven_context(limits) or 4096
        for kv in _working_kv_dtypes(limits, max_ctx):
            serve[f"kv_{kv}"] = {
                "max_model_len": max_ctx,
                "kv_cache_dtype": kv,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": gpu_mem_util,
            }
        bench["agentic"] = {
            "random_input_len": int(max_ctx * 0.75),
            "random_output_len": 512,
            "max_concurrency": 1,
            "num_prompts": 10,
        }

    elif profile == "creative":
        max_ctx = _max_proven_context(limits) or 4096
        for kv in _working_kv_dtypes(limits, max_ctx):
            for seqs in [1, 2, 4, 8]:
                serve[f"kv_{kv}-seqs{seqs}"] = {
                    "max_model_len": max_ctx,
                    "kv_cache_dtype": kv,
                    "max_num_seqs": seqs,
                    "gpu_memory_utilization": gpu_mem_util,
                }
        for ol in [1024, 2048, 4096]:
            bench[f"out{ol}"] = {
                "random_input_len": 256,
                "random_output_len": ol,
                "max_concurrency": 1,
                "num_prompts": 5,
            }

    serve_path = output_dir / "serve_params.json"
    bench_path = output_dir / "bench_params.json"
    serve_path.write_text(json.dumps(serve, indent=2))
    bench_path.write_text(json.dumps(bench, indent=2))

    return serve_path, bench_path, subcommand


def _load_summaries(experiment_dir: Path) -> list[dict]:
    rows = []
    for summary in sorted(experiment_dir.glob("**/summary.json")):
        with open(summary) as f:
            data = json.load(f)
        combo_name = summary.parent.name
        if isinstance(data, list):
            for run in data:
                run["_combo_dir"] = combo_name
                rows.append(run)
        else:
            data["_combo_dir"] = combo_name
            rows.append(data)
    return rows


def pick_winner(experiment_dir: Path, profile: str) -> tuple[dict, list[dict]]:
    rows = _load_summaries(experiment_dir)
    if not rows:
        print(f"ERROR: No results in {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    meta = PROFILES[profile]
    metric = meta["metric"]
    maximize = meta["direction"] == "maximize"

    combos: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        combos[r["_combo_dir"]].append(r)

    scored = []
    for name, runs in combos.items():
        valid = [r for r in runs if float(r.get("failed", 0)) == 0]
        if not valid:
            continue
        mean_val = sum(float(r[metric]) for r in valid) / len(valid)
        scored.append({
            "name": name,
            metric: round(mean_val, 1),
            "mean_ttft_ms": round(sum(float(r.get("mean_ttft_ms", 0)) for r in valid) / len(valid), 1),
            "median_tpot_ms": round(sum(float(r.get("median_tpot_ms", 0)) for r in valid) / len(valid), 1),
            "output_throughput": round(sum(float(r.get("output_throughput", 0)) for r in valid) / len(valid), 1),
            "params": valid[0],
        })

    if not scored:
        print(f"ERROR: No valid results in {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    scored.sort(key=lambda x: x[metric], reverse=maximize)
    return scored[0], scored[:3]
