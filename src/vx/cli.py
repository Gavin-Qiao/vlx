"""vx — vLLM model manager CLI."""


import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vx.config import (
    MODELS_DIR,
    limits_path,
    profile_path,
    read_limits,
)
from vx.models import scan_models, fuzzy_match, ModelInfo

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
        from vx.gpu import get_gpu_info
        gpu = get_gpu_info()
        gpu_line = f"{gpu.name} ({gpu.vram_total_gb:.0f} GB, CUDA {gpu.cuda})"
    except Exception:
        gpu_line = "[dim]unavailable[/dim]"

    models = scan_models(MODELS_DIR)
    probed = sum(
        1 for m in models if read_limits(limits_path(m.provider, m.model_name))
    )

    lines = [
        f"  [bold]GPU[/bold]       {gpu_line}",
        f"  [bold]Models[/bold]    {len(models)} downloaded, {probed} probed",
        "",
        "  [bold]Commands[/bold]",
        "    download    Search & download from HuggingFace",
        "    models      List models with limits & benchmarks",
        "    tune        Probe limits + benchmark a model",
        "    start       Start serving",
        "    stop        Stop serving",
        "    compare     Find the best model for a workload",
    ]
    console.print(Panel("\n".join(lines), title="[bold cyan]vLLM[/bold cyan]", border_style="cyan"))


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
def download(model_id: str = typer.Argument(None, help="HuggingFace model ID")):
    """Search & download a model from HuggingFace."""
    console.print("[dim]Download not yet implemented.[/dim]")


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
    from vx.gpu import get_gpu_info, compute_gpu_memory_utilization
    from vx.config import BENCHMARKS_DIR, VLLM_BIN, write_limits, write_profile_yaml
    from vx.probe import probe_model
    from vx.tune import generate_sweep_params, pick_winner, PROFILES
    from vx.models import quant_flag as get_quant_flag

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

    console.print(f"\n[bold]Jobs: {total_jobs}[/bold]  ({len(models_to_tune)} models)\n")

    for m, job_type in jobs:
        model_key = str(m.path)

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
                write_limits(lim_path_w, limits_data)
                current_model_limits[model_key] = limits_data
                console.print(f"  [green]✓ Limits saved[/green] ({_elapsed()} elapsed)\n")

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

        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
        exp_name = f"{prof}-{timestamp}"
        exp_dir = BENCHMARKS_DIR / m.model_name / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        serve_p, bench_p, subcommand = generate_sweep_params(
            profile=prof, limits=limits_data, output_dir=exp_dir,
        )

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
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            from rich.progress import TimeElapsedColumn, TimeRemainingColumn

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                TextColumn("[dim]{task.fields[detail]}[/dim]"),
                console=console,
                refresh_per_second=1,
            ) as progress:
                header_task = progress.add_task(
                    f"[{completed_jobs}/{total_jobs}] {m.model_name} → {prof}",
                    total=total_results,
                    detail="starting...",
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
                        )
                        _time.sleep(3)
                    result_files = list(exp_dir.glob("**/run=*.json"))
                    progress.update(tid, completed=len(result_files), detail="done")  # type: ignore[arg-type]

                poller = threading.Thread(target=_poll_results, daemon=True)
                poller.start()
                proc.wait()
                poller.join(timeout=5)

            if proc.returncode == 0:
                break
            if attempt < 2:
                console.print("  [yellow]Retrying with --resume...[/yellow]")

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
            console.print(f"  [green]✓ {prof} config written[/green]\n")
            passed += 1
        except SystemExit:
            console.print(f"  [yellow]✗ No results for {prof}[/yellow]\n")
            failed += 1

        completed_jobs += 1

    console.print(f"[bold green]Done.[/bold green]  {passed} passed, {failed} failed, {_elapsed()} elapsed")


@app.command()
def start(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    profile: str = typer.Option("interactive", "--profile", "-p", help="Profile to use"),
):
    """Start serving a model."""
    console.print("[dim]Start not yet implemented.[/dim]")


@app.command()
def stop():
    """Stop the vLLM service."""
    console.print("[dim]Stop not yet implemented.[/dim]")


@app.command()
def compare(
    users: int = typer.Option(None, "--users", "-u", help="Number of concurrent users"),
    use_case: str = typer.Option(None, "--use-case", help="interactive, batch, agentic, creative"),
    context: int = typer.Option(None, "--context", "-c", help="Required context length in tokens"),
):
    """Find the best model for a workload."""
    console.print("[dim]Compare not yet implemented.[/dim]")
