"""vx — vLLM model manager CLI."""


import pathlib
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vlx.config import (
    MODELS_DIR,
    limits_path,
    profile_path,
    read_limits,
    read_timing,
    write_timing,
)
from vlx.models import scan_models, fuzzy_match, ModelInfo

app = typer.Typer(help="vLLM model manager")
console = Console()

PROFILE_NAMES = ["interactive", "batch", "agentic", "creative"]


def _resolve_model(query: str) -> ModelInfo:
    models = scan_models(MODELS_DIR)
    if not models:
        console.print("[red]No models found.[/red] Run: vx download")
        raise typer.Exit(1)
    matches = fuzzy_match(query, models)
    if len(matches) == 0:
        console.print(f"[red]No model matching '{query}'[/red]")
        for m in models:
            console.print(f"  {m.full_name}")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[yellow]'{query}' matches multiple models:[/yellow]")
        for m in matches:
            console.print(f"  {m.full_name}")
        raise typer.Exit(1)
    return matches[0]


@app.callback(invoke_without_command=True)
def dashboard(ctx: typer.Context):
    """Show status dashboard when called with no subcommand."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        from vlx.gpu import get_gpu_info
        gpu = get_gpu_info()
        gpu_line = f"{gpu.name} ({gpu.vram_total_gb:.0f} GB, CUDA {gpu.cuda})"
    except Exception:
        gpu_line = "[dim]unavailable[/dim]"

    from vlx.serve import is_vllm_running
    from vlx.config import active_yaml_path, read_profile_yaml

    models = scan_models(MODELS_DIR)
    probed = sum(
        1 for m in models if read_limits(limits_path(m.provider, m.model_name))
    )

    serving_line = "[dim]not running[/dim]"
    if is_vllm_running():
        active = active_yaml_path()
        if active.exists() and active.is_symlink():
            cfg = read_profile_yaml(active)
            model_path = cfg.get("model", "?") if cfg else "?"
            model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path
            port = cfg.get("port", 8888) if cfg else 8888
            serving_line = f"[green]{model_name}[/green] at :{port}"
        else:
            serving_line = "[green]active[/green]"

    lines = [
        f"  [bold]GPU[/bold]       {gpu_line}",
        f"  [bold]Models[/bold]    {len(models)} downloaded, {probed} probed",
        f"  [bold]Serving[/bold]   {serving_line}",
        "",
    ]

    from rich.table import Table as _Tbl
    from rich.text import Text as _Txt
    cmd_tbl = _Tbl(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    cmd_tbl.add_column(min_width=26)
    cmd_tbl.add_column(style="dim")
    for cmd, desc in [
        ("start [model] [profile]", "Start serving (interactive picker if no args)"),
        ("stop", "Stop the vLLM service"),
        ("status", "Show what's currently serving"),
        ("models", "List models with limits & profiles"),
        ("tune [model] [profile]", "Probe limits + benchmark"),
        ("doctor", "Check system readiness"),
    ]:
        cmd_tbl.add_row(_Txt(cmd, style="bold cyan"), desc)

    from rich.console import Group
    body = Group(
        "\n".join(lines) + "\n  [bold]Commands[/bold]",
        cmd_tbl,
    )
    console.print(Panel(body, title="[bold cyan]vx[/bold cyan]", border_style="cyan"))


@app.command()
def models(model: str = typer.Argument(None, help="Model name for detail view")):
    """List downloaded models."""
    all_models = scan_models(MODELS_DIR)
    if not all_models:
        console.print("[dim]No models found.[/dim] Run: vx download")
        return

    if model:
        m = _resolve_model(model)
        _show_model_detail(m)
        return

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Probed")
    table.add_column("Tuned")
    table.add_column("Max Context", justify="right")

    for m in all_models:
        lim = read_limits(limits_path(m.provider, m.model_name))
        probed = "\u2713" if lim else "\u2717"

        tuned_profiles = [
            p for p in PROFILE_NAMES
            if profile_path(m.provider, m.model_name, p).exists()
        ]
        tuned = ", ".join(tuned_profiles) if tuned_profiles else "\u2717"

        max_ctx = "\u2014"
        if lim:
            for ctx_str in sorted(lim["limits"].keys(), key=int, reverse=True):
                entry = lim["limits"][ctx_str]
                if any(v is not None for v in entry.values()):
                    max_ctx = f"{int(ctx_str) // 1024}k"
                    break

        table.add_row(m.full_name, f"{m.model_size_gb} GB", probed, tuned, max_ctx)

    console.print(table)


def _show_model_detail(m: ModelInfo):
    lim = read_limits(limits_path(m.provider, m.model_name))

    console.print(f"\n[bold]{m.full_name}[/bold]")
    console.print(f"  Arch: {m.architecture}  Quant: {m.quant_method or 'none'}  MoE: {m.is_moe}")
    console.print(f"  Size: {m.model_size_gb} GB  Max positions: {m.max_position_embeddings}\n")

    if not lim:
        console.print(f"  [dim]Not probed yet.[/dim] Run: vx tune {m.model_name.lower()}\n")
        return

    table = Table(title="Context / Concurrency Limits")
    table.add_column("Context", justify="right")
    table.add_column("kv auto", justify="right")
    table.add_column("kv fp8", justify="right")

    for ctx_str in sorted(lim["limits"].keys(), key=int):
        entry = lim["limits"][ctx_str]
        auto = str(entry.get("auto")) if entry.get("auto") is not None else "OOM"
        fp8 = str(entry.get("fp8")) if entry.get("fp8") is not None else "OOM"
        table.add_row(f"{int(ctx_str) // 1024}k", auto, fp8)

    console.print(table)

    tuned = [p for p in PROFILE_NAMES if profile_path(m.provider, m.model_name, p).exists()]
    if tuned:
        console.print(f"\n  Tuned profiles: {', '.join(tuned)}")
    else:
        console.print("\n  [dim]No profiles tuned yet.[/dim]")
    console.print()


@app.command()
def download(model_id: str = typer.Argument(None, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-27B-FP8)")):
    """Download a model from HuggingFace to the models directory."""
    from huggingface_hub import snapshot_download, HfApi
    from vlx.config import cfg as _cfg

    models_dir = _cfg().models_dir

    if model_id is None:
        model_id = typer.prompt("HuggingFace model ID (e.g. Qwen/Qwen3.5-27B-FP8)")

    if "/" not in model_id:
        console.print(f"[yellow]Searching HuggingFace for '{model_id}'...[/yellow]")
        api = HfApi()
        results = list(api.list_models(search=model_id, sort="downloads", limit=10))
        if not results:
            console.print(f"[red]No models found for '{model_id}'.[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Results for '{model_id}':[/bold]")
        for i, m in enumerate(results, 1):
            downloads = f"{m.downloads:,}" if m.downloads else "?"
            console.print(f"  {i}) {m.id}  [dim]({downloads} downloads)[/dim]")

        choice = _pick_number("\nModel number", len(results))
        model_id = results[choice - 1].id

    # provider/model_name → local path
    parts = model_id.split("/")
    if len(parts) != 2:
        console.print(f"[red]Invalid model ID '{model_id}'.[/red] Expected format: provider/model-name")
        raise typer.Exit(1)

    provider, model_name = parts
    local_dir = models_dir / provider / model_name

    console.print(f"\n[bold]Downloading[/bold] {model_id}")
    console.print(f"  To: {local_dir}\n")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
        )
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        raise typer.Exit(1)

    from vlx.models import detect_model
    info = detect_model(local_dir)
    if info:
        console.print(f"\n[green]Downloaded {info.full_name}[/green]")
        console.print(f"  Size: {info.model_size_gb:.1f} GB")
        console.print(f"  Quant: {info.quant_method or 'none'}")
        console.print(f"  Context: {info.max_position_embeddings:,} tokens")
        console.print(f"\nNext: vlx tune {model_name}")
    else:
        console.print(f"\n[green]Downloaded to {local_dir}[/green]")


@app.command()
def tune(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    profile: str = typer.Argument(None, help="Profile: interactive, batch, agentic, creative, all"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show commands without running"),
    reprobe: bool = typer.Option(False, "--reprobe", help="Force re-probe even if cached"),
    no_probe: bool = typer.Option(False, "--no-probe", help="Skip probe phase"),
    all_models: bool = typer.Option(False, "--all", help="Tune all downloaded models"),
):
    """Probe limits + benchmark a model."""
    import subprocess
    from datetime import datetime
    from vlx.gpu import get_gpu_info, compute_gpu_memory_utilization
    from vlx.config import BENCHMARKS_DIR, VLLM_BIN, write_limits, write_profile_yaml
    from vlx.probe import probe_model
    from vlx.tune import generate_sweep_params, pick_winner, PROFILES
    from vlx.models import quant_flag as get_quant_flag

    # Resolve which models to tune
    if all_models:
        models_to_tune = scan_models(MODELS_DIR)
        if not profile:
            profile = "all"
    elif model:
        m = _resolve_model(model)
        models_to_tune = [m]
    else:
        all_m = scan_models(MODELS_DIR)
        if not all_m:
            console.print("[red]No models found.[/red]")
            raise typer.Exit(1)
        for i, m in enumerate(all_m):
            console.print(f"  {i + 1}) {m.full_name}")
        console.print(f"  {len(all_m) + 1}) All models")
        idx = typer.prompt("Which model?", type=int) - 1
        if idx == len(all_m):
            models_to_tune = all_m
        else:
            models_to_tune = [all_m[idx]]

    # Resolve which profiles to run
    if not profile:
        choices = list(PROFILES.keys()) + ["all", "skip"]
        labels = {
            "interactive": "Interactive (chat, coding, RAG)",
            "batch": "Batch processing",
            "agentic": "Agentic workflows",
            "creative": "Creative writing",
            "all": "All profiles",
            "skip": "Skip benchmarks (probe only)",
        }
        console.print("\n[bold]What's this model for?[/bold]")
        for i, c in enumerate(choices):
            console.print(f"  {i + 1}) {labels.get(c, c)}")
        idx = typer.prompt("Choice", type=int) - 1
        profile = choices[idx]

    profiles_to_run = list(PROFILES.keys()) if profile == "all" else ([] if profile == "skip" else [profile])

    # Clear the interactive menu noise — from here on it's just progress
    console.clear()

    gpu = get_gpu_info()
    gpu_mem_util = compute_gpu_memory_utilization(gpu.vram_total_gb)

    # Build job list: (model, phase) pairs for progress tracking
    import time as _time
    jobs: list[tuple[ModelInfo, str]] = []  # (model, "probe" | profile_name)
    for m in models_to_tune:
        lim_path_check = limits_path(m.provider, m.model_name)
        existing_check = read_limits(lim_path_check)
        if not existing_check or reprobe:
            if not no_probe:
                jobs.append((m, "probe"))
        for prof in profiles_to_run:
            jobs.append((m, prof))

    total_jobs = len(jobs)
    completed_jobs = 0
    start_time = _time.monotonic()
    passed = 0
    failed = 0

    def _fmt_duration(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        elif s < 3600:
            return f"{s // 60}m {s % 60}s"
        elif s < 86400:
            h = s // 3600
            m = (s % 3600) // 60
            return f"{h}h {m}m {s % 60}s"
        else:
            d = s // 86400
            h = (s % 86400) // 3600
            m = (s % 3600) // 60
            return f"{d}d {h}h {m}m"

    def _eta() -> str:
        if completed_jobs == 0:
            return "estimating..."
        elapsed = _time.monotonic() - start_time
        avg = elapsed / completed_jobs
        remaining = avg * (total_jobs - completed_jobs)
        return _fmt_duration(remaining)

    def _elapsed() -> str:
        return _fmt_duration(_time.monotonic() - start_time)

    def _status_line(label: str) -> str:
        return f"[{completed_jobs}/{total_jobs}]  {label}  |  elapsed {_elapsed()}  |  ETA {_eta()}"

    if total_jobs == 0:
        console.print("[dim]Nothing to do.[/dim]")
        return

    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    import json as _json
    import threading

    current_model_limits: dict[str, dict] = {}
    current_job_label = ""
    Text("")

    def _render_header() -> Text:
        header = Text()
        header.append(f"[{completed_jobs}/{total_jobs}]", style="bold")
        header.append(f"  {current_job_label}", style="cyan")
        header.append(f"  |  elapsed {_elapsed()}  |  ETA {_eta()}", style="dim")
        return header

    if total_jobs == 0:
        console.print("[dim]Nothing to do.[/dim]")
        return

    # Show time estimate from historical data
    all_timings = [read_timing(m.provider, m.model_name) for m, _ in jobs]
    has_history = any(t for t in all_timings)
    if has_history:
        est_total = 0.0
        for (m, job_type), timing in zip(jobs, all_timings):
            if job_type in timing:
                est_total += timing[job_type]
            elif timing:
                # Use average of known phases as fallback
                avg = sum(timing.values()) / len(timing)
                est_total += avg
        if est_total > 0:
            console.print(
                f"\n[bold]Jobs: {total_jobs}[/bold]  ({len(models_to_tune)} models)"
                f"  [dim]— estimated {_fmt_duration(est_total)} based on previous runs[/dim]\n"
            )
        else:
            console.print(f"\n[bold]Jobs: {total_jobs}[/bold]  ({len(models_to_tune)} models)\n")
    else:
        console.print(f"\n[bold]Jobs: {total_jobs}[/bold]  ({len(models_to_tune)} models)\n")

    for m, job_type in jobs:
        model_key = str(m.path)
        job_t0 = _time.monotonic()

        if job_type == "probe":
            current_job_label = f"{m.model_name} → probe"

            if dry_run:
                console.print(f"[bold cyan]{_render_header()}[/bold cyan]")
                console.print("  [dim]DRY RUN: would probe[/dim]\n")
                current_model_limits[model_key] = {"limits": {}, "gpu_memory_utilization": gpu_mem_util}
            else:
                probe_detail = Text("  Starting probe...", style="dim")

                class _ProbeDisplay:
                    """Renderable that recomputes header (with live elapsed) on every refresh."""
                    def __rich_console__(self, console_obj: object, options: object):  # type: ignore[override]
                        from rich.console import Group
                        yield from Group(_render_header(), probe_detail).__rich_console__(console_obj, options)  # type: ignore[arg-type]

                def _on_probe_step(ctx: int, kv_dtype: str, phase: str, result: object) -> None:
                    ctx_k = f"{ctx // 1024}k"
                    probe_detail.truncate(0)
                    if phase == "context" and result is None:
                        probe_detail.append(f"  Testing {ctx_k} kv={kv_dtype}...", style="dim")
                    elif phase == "context" and result is True:
                        probe_detail.append(f"  {ctx_k} kv={kv_dtype} ", style="dim")
                        probe_detail.append("✓ fits", style="green")
                    elif phase == "context" and result is False:
                        probe_detail.append(f"  {ctx_k} kv={kv_dtype} ", style="dim")
                        probe_detail.append("✗ OOM", style="red")
                    elif phase == "concurrency" and result is None:
                        probe_detail.append(f"  {ctx_k} kv={kv_dtype} → testing concurrency...", style="dim")
                    elif phase == "concurrency":
                        probe_detail.append(f"  {ctx_k} kv={kv_dtype} → ", style="dim")
                        probe_detail.append(f"{result} slots", style="bold green" if result else "red")

                with Live(_ProbeDisplay(), console=console, refresh_per_second=1) as live:
                    def _refresh_live(*a: object) -> None:
                        _on_probe_step(*a)  # type: ignore[arg-type]
                        live.refresh()

                    limits_data = probe_model(
                        model_info=m, gpu_mem_util=gpu_mem_util, vram_total_gb=gpu.vram_total_gb,
                        on_step=_refresh_live,
                    )

                lim_path_w = limits_path(m.provider, m.model_name)
                has_working = any(
                    v is not None
                    for entry in limits_data.get("limits", {}).values()
                    for v in entry.values()
                )
                job_dur = _time.monotonic() - job_t0
                if has_working:
                    write_limits(lim_path_w, limits_data)
                    current_model_limits[model_key] = limits_data
                    write_timing(m.provider, m.model_name, "probe", job_dur)
                    console.print(f"  [green]✓ Limits saved[/green] ({_fmt_duration(job_dur)} elapsed)")

                    # Preheat flashinfer JIT cache for the service user
                    from vlx.probe import preheat_jit
                    console.print("  [dim]Preheating JIT cache for service user...[/dim]")
                    for kv in ("auto", "fp8"):
                        ok = preheat_jit(model_info=m, gpu_mem_util=gpu_mem_util, kv_dtype=kv)
                        tag = "[green]✓[/green]" if ok else "[red]✗[/red]"
                        console.print(f"    kv={kv} {tag}")
                    console.print()
                else:
                    current_model_limits[model_key] = limits_data
                    console.print(f"  [yellow]⚠ No working configs found — limits not cached[/yellow] ({_fmt_duration(job_dur)} elapsed)\n")

            completed_jobs += 1
            passed += 1
            continue

        # Benchmark job — need limits data
        if model_key not in current_model_limits:
            lim_path_r = limits_path(m.provider, m.model_name)
            existing_lim = read_limits(lim_path_r)
            if existing_lim:
                current_model_limits[model_key] = existing_lim
            else:
                console.print(f"  [red]No probe data for {m.model_name}, skipping {job_type}[/red]\n")
                completed_jobs += 1
                failed += 1
                continue

        # Check if model has any working configs at all
        limits_data = current_model_limits[model_key]
        has_any = any(
            v is not None
            for entry in limits_data.get("limits", {}).values()
            for v in entry.values()
        )
        if not has_any:
            console.print(
                f"[yellow]  [{completed_jobs}/{total_jobs}] {m.model_name} → {job_type}  |  elapsed {_elapsed()}[/yellow]"
            )
            console.print("  [red]✗ Skipped — model has no working configs (all OOM during probe)[/red]\n")
            completed_jobs += 1
            failed += 1
            continue

        prof = job_type
        current_job_label = f"{m.model_name} → {prof}"

        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        exp_name = f"{prof}-{timestamp}"
        exp_dir = BENCHMARKS_DIR / m.model_name / exp_name
        # Write params to a temp dir — vllm bench sweep creates exp_dir itself
        import tempfile as _tmpmod
        from pathlib import Path as _Path
        _params_dir = _Path(_tmpmod.mkdtemp(prefix="vx-params-"))

        serve_p, bench_p, subcommand = generate_sweep_params(
            profile=prof, limits=limits_data, output_dir=_params_dir,
        )
        (BENCHMARKS_DIR / m.model_name).mkdir(parents=True, exist_ok=True)

        base_serve = f"{VLLM_BIN} serve {m.path} --dtype bfloat16 --trust-remote-code"
        qf = get_quant_flag(m.quant_method)
        if qf:
            base_serve += f" {qf}"

        sweep_cmd = [
            str(VLLM_BIN), "bench", "sweep", subcommand,
            "--serve-cmd", base_serve,
            "--bench-cmd", f"{VLLM_BIN} bench serve --dataset-name random",
            "--serve-params", str(serve_p),
            "--bench-params", str(bench_p),
            "--num-runs", "3",
            "--server-ready-timeout", "600",
            "--output-dir", str(BENCHMARKS_DIR / m.model_name),
            "--experiment-name", exp_name,
        ]

        if dry_run:
            console.print(f"[bold cyan]{_render_header()}[/bold cyan]")
            console.print(f"  [dim]DRY RUN: {' '.join(sweep_cmd)}[/dim]\n")
            completed_jobs += 1
            passed += 1
            continue

        serve_count = len(_json.loads(serve_p.read_text()))
        bench_count = len(_json.loads(bench_p.read_text()))
        total_combos = serve_count * bench_count
        num_runs = 3
        total_results = total_combos * num_runs

        for attempt in range(3):
            cmd = sweep_cmd + (["--resume"] if attempt > 0 else [])
            import tempfile as _tmpfile
            _stderr_f = _tmpfile.NamedTemporaryFile(
                prefix="vx-bench-", suffix=".stderr", delete=False,
            )
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=_stderr_f)

            from rich.progress import TimeElapsedColumn

            bench_t0 = _time.monotonic()

            def _fmt_eta() -> str:
                elapsed = _time.monotonic() - bench_t0
                n = len(list(exp_dir.glob("**/run=*.json"))) if exp_dir.exists() else 0
                if n < 1 or elapsed < 5:
                    return "ETA -:--:--"
                remaining = elapsed / n * (total_results - n)
                return f"ETA {_fmt_duration(remaining)}"

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TextColumn("{task.fields[eta]}"),
                TextColumn("[dim]{task.fields[detail]}[/dim]"),
                console=console,
                refresh_per_second=1,
            ) as progress:
                header_task = progress.add_task(
                    f"[{completed_jobs}/{total_jobs}] {m.model_name} → {prof}",
                    total=total_results,
                    detail="starting...",
                    eta="ETA -:--:--",
                )

                def _poll_results(tid: object = header_task) -> None:
                    while proc.poll() is None:
                        result_files = list(exp_dir.glob("**/run=*.json"))
                        count = len(result_files)
                        last = result_files[-1].parent.name if result_files else "waiting..."
                        progress.update(
                            tid,  # type: ignore[arg-type]
                            completed=count,
                            detail=last,
                            eta=_fmt_eta(),
                        )
                        _time.sleep(3)
                    result_files = list(exp_dir.glob("**/run=*.json"))
                    progress.update(tid, completed=len(result_files), detail="done", eta=_fmt_eta())  # type: ignore[arg-type]

                poller = threading.Thread(target=_poll_results, daemon=True)
                poller.start()
                proc.wait()
                poller.join(timeout=5)

            # Log stderr on failure
            try:
                with open(_stderr_f.name) as _sf:
                    _bench_stderr = _sf.read()
                if proc.returncode != 0 and _bench_stderr.strip():
                    _stderr_tail = _bench_stderr[-2000:]
                    from vlx.config import LOGS_DIR
                    LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    _log_name = LOGS_DIR / f"bench-{m.model_name}-{prof}-{attempt}.log"
                    _log_name.write_text(_bench_stderr)
                    console.print(f"  [dim]stderr saved to {_log_name}[/dim]")
            except OSError:
                pass
            finally:
                try:
                    import os as _os2
                    _os2.unlink(_stderr_f.name)
                except OSError:
                    pass

            if proc.returncode == 0:
                break
            if attempt < 2:
                console.print("  [yellow]Retrying with --resume...[/yellow]")

        job_dur = _time.monotonic() - job_t0
        try:
            winner, top3 = pick_winner(exp_dir, profile=prof)
            config_data = {
                "model": str(m.path),
                "host": "0.0.0.0",
                "port": 8888,
                "dtype": "bfloat16",
                "trust-remote-code": True,
                "gpu-memory-utilization": gpu_mem_util,
            }
            for key in ["kv_cache_dtype", "enable_prefix_caching", "max_model_len", "max_num_seqs", "max_num_batched_tokens"]:
                val = winner["params"].get(key)
                if val is not None:
                    config_data[key.replace("_", "-")] = val

            cfg_path = profile_path(m.provider, m.model_name, prof)
            write_profile_yaml(cfg_path, config_data, comment=f"vx tune — {prof} profile\nBenchmark: {exp_dir}")
            write_timing(m.provider, m.model_name, prof, job_dur)
            console.print(f"  [green]✓ {prof} config written[/green]\n")
            passed += 1
        except SystemExit:
            console.print(f"  [yellow]✗ No results for {prof}[/yellow]\n")
            failed += 1

        completed_jobs += 1

    console.print(f"[bold green]Done.[/bold green]  {passed} passed, {failed} failed, {_elapsed()} elapsed")


def _launch_vllm(cfg_path: "pathlib.Path", label: str) -> None:
    """Stop any running vLLM, start with given config, wait for health."""
    from vlx.config import read_profile_yaml
    from vlx.serve import start_vllm, stop_vllm, is_vllm_running
    import time

    cfg = read_profile_yaml(cfg_path) or {}

    if is_vllm_running():
        console.print("[yellow]vLLM is already running. Stopping first...[/yellow]")
        stop_vllm()
        time.sleep(2)

    ctx = cfg.get("max-model-len", "?")
    ctx_d = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
    console.print(f"\n[bold]Starting[/bold] with [cyan]{label}[/cyan]")
    console.print(f"  context: {ctx_d}  kv: {cfg.get('kv-cache-dtype', 'auto')}"
                  f"  seqs: {cfg.get('max-num-seqs', '?')}")

    start_vllm(cfg_path)

    from urllib.request import urlopen
    port = cfg.get("port", 8888)
    health_url = f"http://localhost:{port}/health"

    console.print(f"  [dim]Waiting for {health_url} ...[/dim]")
    for i in range(150):  # 300s max
        time.sleep(2)
        # Check health endpoint directly — the only reliable signal
        try:
            resp = urlopen(health_url, timeout=2)
            if resp.status == 200:
                console.print(f"\n[bold green]vLLM is running[/bold green] at http://localhost:{port}/v1")
                console.print(f"  Config: {cfg_path}")
                from vlx.config import cfg as _cfg
                svc = _cfg().service_name
                console.print(f"  Logs:   sudo journalctl -u {svc} -f\n")
                return
        except Exception:
            pass
        # Detect crash-loop early — if systemd gave up, no point waiting
        if i > 5 and not is_vllm_running():
            from vlx.config import cfg as _cfg
            svc = _cfg().service_name
            console.print(f"[red]Service stopped unexpectedly.[/red] Check: sudo journalctl -u {svc} --no-pager -n 50")
            raise typer.Exit(1)
        if i > 0 and i % 15 == 0:
            console.print(f"  [dim]still starting... ({i * 2}s)[/dim]")
    from vlx.config import cfg as _cfg
    svc = _cfg().service_name
    console.print(f"[red]Timed out waiting for health endpoint.[/red] Check: sudo journalctl -u {svc} --no-pager -n 50")
    raise typer.Exit(1)


def _pick_number(prompt: str, max_val: int) -> int:
    choice = typer.prompt(prompt)
    try:
        n = int(choice)
        if 1 <= n <= max_val:
            return n
    except ValueError:
        pass
    console.print("[red]Invalid choice.[/red]")
    raise typer.Exit(1)


def _custom_config(m: ModelInfo) -> "pathlib.Path":
    """Guide user through manual parameter selection, write a temp config."""
    from vlx.config import read_limits, write_profile_yaml
    from vlx.gpu import get_gpu_info, compute_gpu_memory_utilization

    lim = read_limits(limits_path(m.provider, m.model_name))
    if not lim:
        console.print(f"[red]No probe data for {m.model_name}.[/red] Run: vx tune {m.model_name}")
        raise typer.Exit(1)

    gpu = get_gpu_info()
    gpu_mem_util = compute_gpu_memory_utilization(gpu.vram_total_gb)
    limits = lim.get("limits", {})

    # 1. Context window
    working_ctxs = sorted(
        (int(c) for c, d in limits.items() if any(v is not None for v in d.values())),
    )
    if not working_ctxs:
        console.print("[red]No working configs found in probe data.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Configure {m.model_name}[/bold]\n")
    console.print("  [bold]1. Context window[/bold] (max tokens per request)")
    for i, ctx in enumerate(working_ctxs, 1):
        console.print(f"     {i}) {ctx // 1024}k")
    ctx_idx = _pick_number("\n  Context", len(working_ctxs))
    chosen_ctx = working_ctxs[ctx_idx - 1]

    # 2. KV cache dtype
    ctx_entry = limits.get(str(chosen_ctx), {})
    working_kvs = [k for k, v in ctx_entry.items() if v is not None]

    console.print("\n  [bold]2. KV cache dtype[/bold]")
    for i, kv in enumerate(working_kvs, 1):
        label = "auto (bfloat16)" if kv == "auto" else "fp8 (saves memory, slightly less accurate)"
        console.print(f"     {i}) {kv}  — {label}")
    kv_idx = _pick_number("\n  KV dtype", len(working_kvs))
    chosen_kv = working_kvs[kv_idx - 1]

    # 3. Concurrent slots
    max_seqs = ctx_entry.get(chosen_kv, 1)
    console.print(f"\n  [bold]3. Concurrent slots[/bold] (max: {max_seqs})")
    console.print("     How many simultaneous requests?")
    seqs_str = typer.prompt(f"  Slots [1-{max_seqs}]", default=str(max_seqs))
    try:
        chosen_seqs = min(max(1, int(seqs_str)), max_seqs)
    except ValueError:
        chosen_seqs = max_seqs

    # 4. Batched tokens (for throughput tuning)
    console.print("\n  [bold]4. Max batched tokens[/bold] (tokens processed per scheduler step)")
    console.print("     Higher = more throughput, uses more memory per step")
    console.print("     1) auto  — let vLLM decide (good for chat)")
    console.print("     2) 2048  — balanced")
    console.print("     3) 4096  — high throughput")
    console.print("     4) 8192  — maximum throughput (batch processing)")
    bt_map = {"1": None, "2": 2048, "3": 4096, "4": 8192}
    bt_choice = typer.prompt("\n  Batched tokens", default="1")
    chosen_bt = bt_map.get(bt_choice)

    # Build config — prefix caching always on (free when unused, wins on repeated prompts)
    cfg: dict = {
        "model": str(m.path),
        "host": "0.0.0.0",
        "port": 8888,
        "dtype": "bfloat16",
        "trust-remote-code": True,
        "gpu-memory-utilization": gpu_mem_util,
        "max-model-len": chosen_ctx,
        "max-num-seqs": chosen_seqs,
        "kv-cache-dtype": chosen_kv,
        "enable-prefix-caching": True,
    }
    if chosen_bt is not None:
        cfg["max-num-batched-tokens"] = chosen_bt

    qf = m.quant_method
    if qf and qf not in ("none", "compressed-tensors"):
        from vlx.models import QUANT_FLAGS
        flag = QUANT_FLAGS.get(qf, "")
        if flag:
            cfg["quantization"] = flag.split()[-1]

    # Summary
    console.print("\n  [bold]Summary[/bold]")
    console.print(f"    Context:        {chosen_ctx // 1024}k")
    console.print(f"    KV dtype:       {chosen_kv}")
    console.print(f"    Slots:          {chosen_seqs}")
    console.print(f"    Batched tokens: {chosen_bt or 'auto'}")
    console.print("    Prefix:         always on")

    confirm = typer.prompt("\n  Start? [Y/n]", default="Y")
    if confirm.strip().lower() == "n":
        raise typer.Exit(0)

    cfg_path = profile_path(m.provider, m.model_name, "custom")
    write_profile_yaml(cfg_path, cfg, comment="vx start — custom config")
    console.clear()
    return cfg_path


@app.command()
def start(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    profile: str = typer.Argument(None, help="Profile: interactive, batch, agentic, creative, custom"),
):
    """Start serving a model with a tuned profile."""
    if model is None:
        # Interactive model picker
        all_models = scan_models(MODELS_DIR)
        if not all_models:
            console.print("[red]No models found.[/red] Run: vx download")
            raise typer.Exit(1)

        console.print("\n[bold]Select a model:[/bold]")
        for i, m in enumerate(all_models, 1):
            tuned = [p for p in PROFILE_NAMES if profile_path(m.provider, m.model_name, p).exists()]
            status = f"[green]{', '.join(tuned)}[/green]" if tuned else "[dim]not tuned[/dim]"
            console.print(f"  {i}) {m.full_name}  ({m.model_size_gb} GB)  {status}")

        choice = _pick_number("\nModel number", len(all_models))
        m = all_models[choice - 1]

        # Profile picker — show tuned profiles + custom option
        tuned = [p for p in PROFILE_NAMES if profile_path(m.provider, m.model_name, p).exists()]
        has_probe = read_limits(limits_path(m.provider, m.model_name)) is not None
        options = list(tuned)
        if has_probe:
            options.append("custom")

        if not options:
            console.print(f"\n[red]No tuned profiles for {m.model_name}.[/red] Run: vx tune {m.model_name}")
            raise typer.Exit(1)

        console.print("\n[bold]Select a profile:[/bold]")
        for i, p in enumerate(options, 1):
            desc = PROFILE_DESCRIPTIONS.get(p, "configure each parameter manually")
            console.print(f"  {i}) {p}  — {desc}")

        pchoice = _pick_number("\nProfile number", len(options))
        profile = options[pchoice - 1]
        console.clear()
    else:
        m = _resolve_model(model)
        if profile is None:
            profile = "interactive"

    if profile == "custom":
        cfg_path = _custom_config(m)
        _launch_vllm(cfg_path, "custom config")
        return

    if profile not in PROFILE_NAMES:
        console.print(f"[red]Unknown profile '{profile}'.[/red] Choose: {', '.join(PROFILE_NAMES + ['custom'])}")
        raise typer.Exit(1)

    cfg_path = profile_path(m.provider, m.model_name, profile)
    if not cfg_path.exists():
        console.print(f"[red]No {profile} profile for {m.model_name}.[/red] Run: vx tune {m.model_name} {profile}")
        raise typer.Exit(1)

    _launch_vllm(cfg_path, f"{profile} profile")


@app.command()
def stop():
    """Stop the vLLM service."""
    from vlx.serve import stop_vllm, is_vllm_running

    if not is_vllm_running():
        console.print("[dim]vLLM is not running.[/dim]")
        return

    stop_vllm()
    console.print("[green]vLLM stopped.[/green]")


@app.command()
def fan(
    mode: str = typer.Argument(None, help="auto | <30-100> | off"),
):
    """GPU fan control — auto curve, fixed speed, or off."""
    from pathlib import Path

    import vlx.fan as _fan
    from vlx.fan import read_state
    from vlx.gpu import set_fan_speed, get_fan_speed
    import os
    import signal as sig
    import time

    _fan._resolve_paths()
    PID_PATH = _fan.PID_PATH
    STATE_PATH = _fan.STATE_PATH

    def _daemon_pid() -> int | None:
        if not PID_PATH.exists():
            return None
        try:
            pid = int(PID_PATH.read_text().strip())
            os.kill(pid, 0)
            # Verify it's actually our daemon, not a recycled PID
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
            if "vlx.fan" not in cmdline:
                PID_PATH.unlink(missing_ok=True)
                STATE_PATH.unlink(missing_ok=True)
                return None
            return pid
        except (ValueError, OSError):
            PID_PATH.unlink(missing_ok=True)
            STATE_PATH.unlink(missing_ok=True)
            return None

    def _stop_daemon() -> bool:
        pid = _daemon_pid()
        if pid is None:
            return False
        os.kill(pid, sig.SIGTERM)
        for _ in range(50):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        PID_PATH.unlink(missing_ok=True)
        STATE_PATH.unlink(missing_ok=True)
        return True

    def _start_daemon(qs: int, qe: int, qm: int) -> None:
        _stop_daemon()
        import subprocess
        import sys
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "vlx.fan",
                "--quiet-start", str(qs),
                "--quiet-end", str(qe),
                "--quiet-max", str(qm),
            ],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        if proc.poll() is not None:
            from vlx.config import cfg as _cfg
            console.print(f"[red]Fan daemon failed to start.[/red] Check {_cfg().logs_dir / 'vx-fan.log'}")
            raise typer.Exit(1)
        console.print(f"[green]Fan daemon started[/green] (pid {proc.pid})")

    def _show_status() -> None:
        speed = get_fan_speed()
        pid = _daemon_pid()
        state = read_state() if pid else None
        try:
            from vlx.fan import _get_gpu_temp
            temp = _get_gpu_temp()
        except Exception:
            temp = None

        temp_str = f"{temp}°C" if temp is not None else "?"
        console.print(f"\n[bold]GPU Fan[/bold]  {speed}% @ {temp_str}")

        if state:
            qs, qe, qm = state["quiet_start"], state["quiet_end"], state["quiet_max"]
            console.print(f"  [green]Auto curve[/green] — quiet {qs:02d}:00-{qe:02d}:00 (max {qm}%), otherwise 100%")
        elif pid:
            console.print("  [green]Auto curve[/green] (defaults)")
        else:
            console.print("  [dim]No daemon — fixed or NVIDIA auto[/dim]")

    # --- Direct mode (non-interactive) ---
    if mode == "auto":
        _start_daemon(9, 18, 60)
        return
    if mode == "off":
        if _stop_daemon():
            console.print("[green]Fan daemon stopped[/green], auto control restored.")
        else:
            console.print("[dim]No fan daemon running.[/dim]")
        return
    if mode is not None:
        try:
            percent = int(mode)
        except ValueError:
            console.print(f"[red]Unknown mode '{mode}'.[/red] Use: auto, off, or 30-100")
            raise typer.Exit(1)
        if not 30 <= percent <= 100:
            console.print("[red]Fan speed must be 30-100.[/red]")
            raise typer.Exit(1)
        _stop_daemon()
        set_fan_speed(percent)
        actual = get_fan_speed()
        console.print(f"[green]Fan set to {percent}%[/green] (reading: {actual}%)")
        return

    # --- Interactive mode ---
    _show_status()

    pid = _daemon_pid()
    state = read_state() if pid else None

    options: list[tuple[str, str]] = []
    if pid:
        options.append(("curve", "Change quiet hours / fan cap"))
        options.append(("fixed", "Fixed speed"))
        options.append(("off", "Off — restore NVIDIA auto"))
    else:
        options.append(("curve", "Auto curve (temp-based with quiet hours)"))
        options.append(("fixed", "Fixed speed"))

    console.print()
    for i, (_, desc) in enumerate(options, 1):
        console.print(f"  {i}) {desc}")

    choice = _pick_number("\nChoice", len(options))
    action = options[choice - 1][0]

    if action == "off":
        _stop_daemon()
        console.print("[green]Fan daemon stopped[/green], auto control restored.")
        return

    if action == "fixed":
        val = typer.prompt("Fan speed % (30-100)", type=int)
        if not 30 <= val <= 100:
            console.print(f"[red]Fan speed must be 30-100, got {val}.[/red]")
            raise typer.Exit(1)
        _stop_daemon()
        set_fan_speed(val)
        console.print(f"[green]Fan set to {val}%[/green]")
        return

    # action == "curve"
    defaults = state or {"quiet_start": 9, "quiet_end": 18, "quiet_max": 60}
    qs = defaults["quiet_start"]
    qe = defaults["quiet_end"]
    qm = defaults["quiet_max"]

    console.print(f"\n  Current: quiet {qs:02d}:00-{qe:02d}:00, max {qm}%")
    qs = typer.prompt("  Quiet start hour (0-23)", type=int, default=qs)
    qe = typer.prompt("  Quiet end hour (0-23)", type=int, default=qe)
    qm = typer.prompt("  Quiet max fan % (30-100)", type=int, default=qm)

    if not 0 <= qs <= 23:
        console.print(f"[red]Quiet start hour must be 0-23, got {qs}.[/red]")
        raise typer.Exit(1)
    if not 0 <= qe <= 23:
        console.print(f"[red]Quiet end hour must be 0-23, got {qe}.[/red]")
        raise typer.Exit(1)
    if not 30 <= qm <= 100:
        console.print(f"[red]Quiet max fan must be 30-100, got {qm}.[/red]")
        raise typer.Exit(1)

    _start_daemon(qs, qe, qm)
    console.print(f"  Quiet {qs:02d}:00-{qe:02d}:00 (max {qm}%), otherwise 100%")


PROFILE_DESCRIPTIONS = {
    "interactive": "Low-latency single-user chat — optimized for fastest time-to-first-token (TTFT)",
    "batch": "Maximum throughput for many requests — optimized for tokens/sec across concurrent users",
    "agentic": "Long-context tool-use / RAG — optimized for TTFT with large input contexts",
    "creative": "Long-form generation (stories, code) — optimized for smooth token-by-token output (TPOT)",
}


@app.command()
def status():
    """Show current vLLM serving status."""
    from vlx.config import active_yaml_path, read_profile_yaml
    from vlx.serve import is_vllm_running

    if not is_vllm_running():
        console.print("[dim]vLLM is not running.[/dim] Start with: vx start")
        return

    active = active_yaml_path()
    if not (active.exists() and active.is_symlink()):
        console.print("[bold green]vLLM is running[/bold green] (no active config link found)")
        return

    target = active.resolve()
    cfg = read_profile_yaml(active) or {}
    model_path = cfg.get("model", "?")
    model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path

    # Detect profile from config filename
    profile_name = "unknown"
    for p in PROFILE_NAMES + ["custom"]:
        if f".{p}.yaml" in target.name:
            profile_name = p
            break

    # Look up model info for max context window
    model_max_ctx = "?"
    try:
        models = scan_models(MODELS_DIR)
        matched = [m for m in models if m.model_name == model_name]
        if matched:
            model_max_ctx = f"{matched[0].max_position_embeddings // 1024}k"
    except Exception:
        pass

    ctx = cfg.get("max-model-len", "?")
    ctx_display = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
    seqs = cfg.get("max-num-seqs", "?")
    kv = cfg.get("kv-cache-dtype", "auto")
    prefix = cfg.get("enable-prefix-caching", False)
    bt = cfg.get("max-num-batched-tokens")
    port = cfg.get("port", 8888)
    gpu_util = cfg.get("gpu-memory-utilization")
    gpu_str = f"{gpu_util:.1%}" if isinstance(gpu_util, float) else str(gpu_util)

    console.print("\n[bold green]vLLM is running[/bold green]")
    from rich.text import Text as _Text
    _model_line = _Text("  ")
    _model_line.append("Model", style="bold")
    _model_line.append(f"      {model_name}  (supports up to {model_max_ctx} context)")
    console.print(_model_line)
    console.print(f"  [bold]Profile[/bold]    {profile_name}")
    console.print(f"  [bold]Endpoint[/bold]   http://localhost:{port}/v1")
    console.print()

    desc = PROFILE_DESCRIPTIONS.get(profile_name, "")
    if desc:
        console.print(f"  [cyan]{desc}[/cyan]")
        console.print()

    console.print("  [bold]Serving config[/bold]")
    console.print(f"    Context window:    {ctx_display}  (max tokens per request, model supports {model_max_ctx})")
    console.print(f"    Concurrent slots:  {seqs}  (max in-flight requests)")
    console.print(f"    KV cache dtype:    {kv}")
    if prefix:
        console.print("    Prefix caching:    enabled  (reuses system prompt KV cache)")
    if bt:
        console.print(f"    Batched tokens:    {bt}")
    console.print(f"    GPU memory:        {gpu_str}")
    console.print(f"\n  [dim]{target}[/dim]\n")


@app.command()
def init():
    """Auto-discover vLLM installation and write config."""
    from vlx.config import (
        CONFIG_FILE, save_config,
        _discover_vllm_root, _discover_cuda_home, _discover_service, _discover_port,
        _build_config, reset_config,
    )

    console.print("\n[bold]Discovering vLLM installation...[/bold]\n")

    root = _discover_vllm_root()
    cuda = _discover_cuda_home()
    svc_name, svc_user = _discover_service()

    if root:
        console.print(f"  vLLM root:     [green]{root}[/green]")
    else:
        console.print("  vLLM root:     [yellow]not found[/yellow] — using /opt/vllm")
        root_input = typer.prompt("  vLLM root path", default="/opt/vllm")
        from pathlib import Path as _Path
        root = _Path(root_input)

    port = _discover_port(root)
    models_count = len(list((root / "models").glob("*/*"))) if (root / "models").exists() else 0

    console.print(f"  Models dir:    {root / 'models'} ({models_count} models)")
    console.print(f"  CUDA:          [green]{cuda}[/green]")
    console.print(f"  Service:       {svc_name} (user: {svc_user})")
    console.print(f"  Port:          {port}")

    if CONFIG_FILE.exists():
        console.print(f"\n  [yellow]Config exists:[/yellow] {CONFIG_FILE}")
        if not typer.confirm("  Overwrite?", default=False):
            console.print("[dim]Aborted.[/dim]")
            return

    config = _build_config(root, cuda, svc_name, svc_user, port)
    path = save_config(config)
    reset_config()

    console.print(f"\n[green]Config written to {path}[/green]\n")


@app.command()
def doctor():
    """Check system readiness for vLLM serving."""
    import subprocess
    import socket
    from pathlib import Path

    from vlx.config import VLLM_BIN, VLLM_ROOT, active_yaml_path, read_profile_yaml, LOGS_DIR
    from vlx.serve import is_vllm_running

    ok_count = 0
    fail_count = 0

    def _ok(msg: str) -> None:
        nonlocal ok_count
        console.print(f"  [green]OK[/green]    {msg}")
        ok_count += 1

    def _fail(msg: str, fix: str = "") -> None:
        nonlocal fail_count
        console.print(f"  [red]FAIL[/red]  {msg}")
        if fix:
            console.print(f"          Fix: {fix}")
        fail_count += 1

    def _warn(msg: str, fix: str = "") -> None:
        console.print(f"  [yellow]WARN[/yellow]  {msg}")
        if fix:
            console.print(f"          Fix: {fix}")

    console.print("\n[bold]vx doctor[/bold]\n")

    # -- Environment --
    console.print("  [bold]Environment[/bold]")

    # nvcc
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5,
                           env={"PATH": "/usr/local/cuda/bin:/usr/bin:/bin"})
        if r.returncode == 0:
            ver = [ln for ln in r.stdout.splitlines() if "release" in ln]
            _ok(f"nvcc {ver[0].split('release')[-1].strip().rstrip(',') if ver else 'found'}")
        else:
            _fail("nvcc not working", "Install CUDA toolkit or check /usr/local/cuda/bin/nvcc")
    except Exception:
        _fail("nvcc not found", "Install CUDA toolkit")

    # vLLM
    try:
        r = subprocess.run([str(VLLM_BIN), "--version"], capture_output=True, text=True, timeout=10)
        _ok(f"vLLM {r.stdout.strip()}")
    except Exception:
        _fail(f"vLLM not found at {VLLM_BIN}")

    # GPU
    try:
        from vlx.gpu import get_gpu_info
        gpu = get_gpu_info()
        mem_used = 0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=5)
            mem_used = int(out.decode().strip().split("\n")[0])
        except Exception:
            pass
        _ok(f"{gpu.name} ({gpu.vram_total_gb:.0f} GB, {mem_used} MiB used)")
    except Exception:
        _fail("GPU not accessible", "Check nvidia-smi")

    # Service user
    from vlx.config import cfg as _cfg
    _c = _cfg()
    try:
        r = subprocess.run(["id", _c.service_user], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok(f"Service user '{_c.service_user}' exists")
        else:
            _fail(f"Service user '{_c.service_user}' not found")
    except Exception:
        _fail("Cannot check service user")

    # systemd unit
    svc_path = Path(f"/etc/systemd/system/{_c.service_name}.service")
    if svc_path.exists():
        svc_content = svc_path.read_text()
        if "ProtectSystem=strict" in svc_content:
            _fail("systemd unit has ProtectSystem=strict",
                  "Remove it — breaks nvcc JIT compilation")
        elif "TimeoutStartSec" not in svc_content:
            _warn("No TimeoutStartSec in service — default 90s may be too short for JIT",
                  "Add TimeoutStartSec=600")
        else:
            _ok("systemd unit configured correctly")
    else:
        _fail(f"No systemd unit at {svc_path}")

    # .env
    env_path = VLLM_ROOT / "configs" / ".env"
    if env_path.exists():
        try:
            env_content = env_path.read_text()
            missing = [v for v in ["CUDA_HOME", "TMPDIR", "VLLM_RPC_BASE_PATH"] if v not in env_content]
            if missing:
                _warn(f".env missing: {', '.join(missing)}")
            else:
                _ok(".env has required variables")
        except PermissionError:
            _ok(".env exists (not readable — OK, contains secrets)")
    else:
        _fail(f"No .env at {env_path}")

    # -- Caches --
    console.print("\n  [bold]Caches[/bold]")

    cache_checks = [
        (VLLM_ROOT / ".cache" / "flashinfer", "FlashInfer JIT"),
        (VLLM_ROOT / ".cache" / "vllm" / "torch_compile_cache", "torch.compile"),
    ]
    for cdir, label in cache_checks:
        if cdir.exists() and any(cdir.rglob("*.so")):
            size_mb = sum(f.stat().st_size for f in cdir.rglob("*") if f.is_file()) / (1024 * 1024)
            _ok(f"{label} cache ({size_mb:.0f} MB)")
        elif cdir.exists():
            _warn(f"{label} cache exists but no compiled .so files — first start will be slow")
        else:
            _warn(f"{label} cache missing — first start will JIT compile (2-10 min)")

    # TMPDIR
    tmp_dir = VLLM_ROOT / "tmp"
    if tmp_dir.exists() and tmp_dir.is_dir():
        _ok(f"TMPDIR at {tmp_dir}")
        # Count stale sockets
        sockets = list(tmp_dir.glob("*"))
        stale = [s for s in sockets if s.is_socket()]
        if len(stale) > 10:
            _warn(f"{len(stale)} stale sockets in {tmp_dir}", "vx cache clean")
    else:
        _fail(f"TMPDIR {tmp_dir} does not exist", f"sudo mkdir -p {tmp_dir} && sudo chown vllm:llm {tmp_dir}")

    # -- Config --
    console.print("\n  [bold]Config[/bold]")

    active = active_yaml_path()
    if active.exists() and active.is_symlink():
        target = active.resolve()
        if target.exists():
            cfg = read_profile_yaml(active) or {}
            model_path = cfg.get("model", "")
            if model_path and Path(model_path).exists():
                _ok(f"active.yaml -> {target.name}")
            else:
                _fail(f"Model path in config does not exist: {model_path}")
        else:
            _fail(f"active.yaml points to missing file: {target}")
    elif active.exists():
        _ok("active.yaml exists (not a symlink)")
    else:
        _warn("No active.yaml — run vx start to configure")

    # Port
    _port = _c.port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        port_in_use = s.connect_ex(("127.0.0.1", _port)) == 0
    if is_vllm_running():
        _ok(f"vLLM serving on port {_port}")
    elif port_in_use:
        _fail(f"Port {_port} in use but vLLM not running — something else is bound")
    else:
        _ok(f"Port {_port} available")

    # Log size
    log_file = LOGS_DIR / "vllm.log"
    if log_file.exists():
        size_mb = log_file.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            _warn(f"vllm.log is {size_mb:.0f} MB", "Consider truncating or adding log rotation")
        else:
            _ok(f"vllm.log ({size_mb:.0f} MB)")

    # Models
    all_models = scan_models(MODELS_DIR)
    probed = sum(1 for m in all_models if read_limits(limits_path(m.provider, m.model_name)))
    _ok(f"{len(all_models)} models downloaded, {probed} probed")

    console.print(f"\n  [bold]{'All clear' if fail_count == 0 else f'{fail_count} issue(s) found'}[/bold]  "
                  f"({ok_count} ok, {fail_count} fail)\n")


