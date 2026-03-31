"""vlx — vLLM model manager CLI."""


import pathlib
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vlx.config import (
    CONFIG_FILE,
    MODELS_DIR,
    limits_path,
    profile_path,
    read_limits,
)
from vlx.models import scan_models, fuzzy_match, ModelInfo
from vlx.lock import VlxLock, LockHeld, notify_user

app = typer.Typer(help="vLLM model manager")
console = Console()



def _lock_or_exit(name: str, description: str) -> VlxLock:
    """Acquire a lock or DM the holder and exit."""
    import os
    lock = VlxLock(name, description)
    try:
        lock.acquire()
    except LockHeld as exc:
        console.print(Panel(
            f"[bold]{exc.message()}[/bold]",
            title=f"[red]vlx: {name} locked[/red]",
            border_style="red",
        ))
        # DM the lock holder
        me = os.environ.get("USER", "?")
        if exc.info and exc.info.user != me:
            notify_user(
                exc.info.user,
                f"{me} is waiting for {name} (you: {exc.info.command})",
            )
        raise typer.Exit(1) from None
    return lock


def _resolve_model(query: str) -> ModelInfo:
    models = scan_models(MODELS_DIR)
    if not models:
        console.print("[red]No models found.[/red] Run: vlx download")
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

    if not CONFIG_FILE.exists():
        console.print(Panel(
            "[bold]First time? Run [cyan]vlx init[/cyan] to set up your system.[/bold]",
            border_style="yellow",
        ))

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
        ("download [model]", "Search & download from HuggingFace"),
        ("start [model]", "Start serving (interactive config picker)"),
        ("stop", "Stop the vLLM service"),
        ("status", "Show what's currently serving"),
        ("models", "List models with limits & profiles"),
        ("tune [model]", "Calculate context & concurrency limits"),
        ("fan [auto|off|30-100]", "GPU fan control with temp curve"),
        ("doctor", "Check system readiness"),
        ("init", "Auto-discover vLLM and write config"),
    ]:
        cmd_tbl.add_row(_Txt(cmd, style="bold cyan"), desc)

    from rich.console import Group
    body = Group(
        "\n".join(lines) + "\n  [bold]Commands[/bold]",
        cmd_tbl,
    )
    console.print(Panel(body, title="[bold cyan]vlx[/bold cyan]", border_style="cyan"))


@app.command()
def models(model: str = typer.Argument(None, help="Model name for detail view")):
    """List downloaded models."""
    all_models = scan_models(MODELS_DIR)
    if not all_models:
        console.print("[dim]No models found.[/dim] Run: vlx download")
        return

    if model:
        m = _resolve_model(model)
        _show_model_detail(m)
        return

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Limits")
    table.add_column("Max Context", justify="right")

    for m in all_models:
        lim = read_limits(limits_path(m.provider, m.model_name))

        max_ctx = "\u2014"
        if lim:
            for ctx_str in sorted(lim["limits"].keys(), key=int, reverse=True):
                entry = lim["limits"][ctx_str]
                if any(v is not None for v in entry.values()):
                    max_ctx = f"{int(ctx_str) // 1024}k"
                    break

        table.add_row(m.full_name, f"{m.model_size_gb} GB", "\u2713" if lim else "\u2717", max_ctx)

    console.print(table)


def _show_model_detail(m: ModelInfo):
    lim = read_limits(limits_path(m.provider, m.model_name))

    console.print(f"\n[bold]{m.full_name}[/bold]")
    console.print(f"  Arch: {m.architecture}  Quant: {m.quant_method or 'none'}  MoE: {m.is_moe}")
    console.print(f"  Size: {m.model_size_gb} GB  Max positions: {m.max_position_embeddings}\n")

    if not lim:
        console.print(f"  [dim]Not probed yet.[/dim] Run: vlx tune {m.model_name.lower()}\n")
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
    console.print()


def _has_gum() -> bool:
    import shutil
    return shutil.which("gum") is not None


def _gum_input(placeholder: str) -> str | None:
    """Interactive text input via gum. Returns None on Esc/Ctrl-C."""
    import subprocess
    r = subprocess.run(
        ["gum", "input", "--placeholder", placeholder, "--width", "60"],
        stdout=subprocess.PIPE, text=True,
    )
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _gum_filter(items: list[str], header: str = "") -> str | None:
    """Interactive fuzzy filter via gum. Returns None on Esc/Ctrl-C."""
    import subprocess
    cmd = ["gum", "filter", "--height", "15", "--placeholder", "Type to filter..."]
    if header:
        cmd.extend(["--header", header])
    r = subprocess.run(cmd, input="\n".join(items), stdout=subprocess.PIPE, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


@app.command()
def download(model_id: str = typer.Argument(None, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-27B-FP8)")):
    """Search and download a model from HuggingFace."""
    from huggingface_hub import snapshot_download, HfApi
    from vlx.config import cfg as _cfg

    models_dir = _cfg().models_dir
    api = HfApi()

    # Check if argument is an existing repo
    if model_id and "/" in model_id:
        try:
            exists = api.repo_exists(model_id, repo_type="model")
        except Exception as e:
            console.print(f"[red]HuggingFace error:[/red] {e}")
            raise typer.Exit(1)
        if exists:
            _download_model(model_id, models_dir, snapshot_download, api)
            return
        # Not found — fall through to keyword search

    use_gum = _has_gum()

    # Search loop
    query = model_id or ""
    results: list = []
    while True:
        if not query:
            if use_gum:
                query = _gum_input("Search HuggingFace (e.g. qwen 27b fp8)") or ""
            else:
                query = typer.prompt("\nSearch HuggingFace (Ctrl-C to quit)")
            if not query:
                # Esc/Ctrl-C in gum input — exit
                return

        # Only re-search if we don't already have results for this query
        if not results:
            console.print(f"[dim]Searching '{query}'...[/dim]")
            try:
                results = list(api.list_models(search=query, sort="downloads", limit=20))
            except Exception as e:
                console.print(f"[red]HuggingFace error:[/red] {e}")
                query = ""
                continue

            if not results:
                console.print(f"[yellow]No results for '{query}'.[/yellow]")
                query = ""
                continue

        if use_gum:
            lines = []
            for m in results:
                dl = f"{m.downloads:,}" if m.downloads else "?"
                lines.append(f"{m.id}  ({dl} downloads)")

            selected = _gum_filter(lines, header=f"Results for '{query}' — Esc to search again")
            if selected is None:
                query = ""
                results = []
                continue

            picked_id = selected.split("  (")[0]
            if _download_model(picked_id, models_dir, snapshot_download, api):
                return
            # User backed out of variant picker — re-show same results
            console.clear()
            continue
        else:
            console.print()
            for i, m in enumerate(results, 1):
                dl = f"{m.downloads:,}" if m.downloads else "?"
                console.print(f"  [bold]{i:>2}[/bold]) {m.id}  [dim]({dl} downloads)[/dim]")

            console.print("\n  [dim]Enter number to download, or type a new search[/dim]")
            answer = typer.prompt("\nChoice or search", default="")
            if not answer:
                continue
            try:
                idx = int(answer)
                if 1 <= idx <= len(results):
                    if _download_model(results[idx - 1].id, models_dir, snapshot_download, api):
                        return
                    # User backed out — re-show same results
                    console.clear()
                    continue
            except ValueError:
                pass
            query = answer
            results = []


def _pick_variants(variants: list) -> list:
    """Interactive variant selection. Returns list of selected Variant objects."""
    import subprocess

    if _has_gum():
        from vlx.variants import format_variant_line
        lines = [format_variant_line(v, index=i) for i, v in enumerate(variants, 1)]
        cmd = [
            "gum", "choose", "--no-limit",
            "--header", "Select variant(s) — space to toggle, enter to confirm:",
            *lines,
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            return []
        selected_lines = r.stdout.strip().split("\n")
        selected = []
        for line in selected_lines:
            for i, v in enumerate(variants, 1):
                if line.lstrip().startswith(f"{i}) "):
                    selected.append(v)
                    break
        return selected
    else:
        prompt = "Select variant(s) (e.g. 1 or 1,2)"
        answer = typer.prompt(prompt, default="")
        if not answer:
            return []
        indices: list[int] = []
        for part in answer.replace(",", " ").split():
            try:
                idx = int(part)
                if 1 <= idx <= len(variants):
                    indices.append(idx)
            except ValueError:
                pass
        return [variants[i - 1] for i in indices]


def _download_model(model_id: str, models_dir: "pathlib.Path", snapshot_download: object, api: object) -> bool:
    """Download a single model by its HuggingFace ID. Returns True if download happened."""
    from vlx.variants import fetch_repo_variants, format_variant_line, _format_bytes

    parts = model_id.split("/")
    if len(parts) != 2:
        console.print(f"[red]Invalid model ID '{model_id}'.[/red] Expected: provider/model-name")
        raise typer.Exit(1)

    provider, model_name = parts
    local_dir = models_dir / provider / model_name

    if local_dir.exists() and any(local_dir.iterdir()):
        console.print(f"[yellow]{model_id} already exists at {local_dir}[/yellow]")
        if not typer.confirm("Re-download?", default=False):
            return True  # not a back-navigation, user chose to keep existing

    # --- Variant picker ---
    console.print(f"\n[bold]{model_id}[/bold]\n")
    console.print("[dim]Fetching file list...[/dim]")

    try:
        variants, shared = fetch_repo_variants(model_id, api)
    except Exception as e:
        console.print(f"[red]Failed to fetch file list:[/red] {e}")
        return False

    if not variants:
        console.print("[yellow]No weight files found in this repo.[/yellow]")
        return False

    # Show shared files summary
    shared_size = sum(shared.values())
    shared_names = ", ".join(list(shared.keys())[:3])
    if len(shared) > 3:
        shared_names += ", ..."
    console.print(f"  Shared: {shared_names} ({_format_bytes(shared_size)}, {len(shared)} files)\n")

    # Show variants
    for i, v in enumerate(variants, 1):
        console.print(format_variant_line(v, index=i))

    console.print()

    # Selection
    selected_variants = _pick_variants(variants)
    if not selected_variants:
        return False  # user backed out — navigate back

    # Confirmation
    total = sum(v.total_bytes for v in selected_variants) + shared_size
    console.print(f"\n  Total download: [bold]{_format_bytes(total)}[/bold]")
    if not typer.confirm("  Download?", default=True):
        return False  # user declined — navigate back

    # Build allow_patterns
    allow_files: list[str] = list(shared.keys())
    for v in selected_variants:
        allow_files.extend(v.files.keys())

    console.print(f"\n[bold]Downloading[/bold] {model_id}")
    console.print(f"  To: {local_dir}\n")

    lock_name = f"download-{provider}--{model_name}"
    lock = VlxLock(lock_name, f"downloading {model_id}")
    try:
        lock.acquire()
    except LockHeld as exc:
        import os
        me = os.environ.get("USER", "?")
        if exc.info and exc.info.user != me:
            notify_user(
                exc.info.user,
                f"{me} is also waiting for {model_id} download",
            )
        console.print(f"[yellow]{model_id} is being downloaded by {exc.info.user if exc.info else 'another user'}.[/yellow]")
        console.print("[dim]Waiting for it to finish...[/dim]")
        from vlx.lock import wait_for_release
        if not wait_for_release(lock_name, timeout=7200):
            console.print("[red]Timed out waiting for download.[/red]")
            raise typer.Exit(1)
        if local_dir.exists() and any(local_dir.iterdir()):
            try:
                from vlx.models import detect_model
                info = detect_model(local_dir)
            except Exception:
                info = None
            if info:
                console.print(f"\n[green]{info.full_name} is ready[/green] (downloaded by {exc.info.user if exc.info else 'another user'})")
            else:
                console.print(f"\n[green]{model_id} is ready.[/green]")
            return True
        console.print("[yellow]Download seems to have failed. Retrying...[/yellow]")
        # Reacquire the lock for the retry
        try:
            lock.acquire()
        except LockHeld:
            console.print("[yellow]Another download started. Exiting.[/yellow]")
            return True

    try:
        snapshot_download(  # type: ignore[operator]
            repo_id=model_id,
            local_dir=local_dir,
            allow_patterns=allow_files,
        )
    except Exception as e:
        console.print(f"[red]Download failed:[/red] {e}")
        raise typer.Exit(1)
    finally:
        lock.release()

    try:
        from vlx.models import detect_model
        info = detect_model(local_dir)
    except Exception:
        info = None
    if info:
        console.print(f"\n[green]Downloaded {info.full_name}[/green]")
        console.print(f"  Size: {info.model_size_gb:.1f} GB")
        console.print(f"  Quant: {info.quant_method or 'none'}")
        console.print(f"  Context: {info.max_position_embeddings:,} tokens")
        console.print(f"\nNext: vlx tune {model_name}")
    else:
        console.print(f"\n[green]Downloaded to {local_dir}[/green]")
    return True


@app.command()
def tune(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    all_models: bool = typer.Option(False, "--all", help="Tune all downloaded models"),
    recalc: bool = typer.Option(False, "--recalc", help="Force recalculation even if cached"),
    gpu_util: float = typer.Option(0.90, "--gpu-util", help="GPU memory utilization (0.5-0.99)"),
):
    """Calculate context and concurrency limits for a model."""
    if not 0.5 <= gpu_util <= 0.99:
        console.print("[red]--gpu-util must be between 0.5 and 0.99[/red]")
        raise typer.Exit(1)

    from vlx.gpu import get_gpu_info
    from vlx.probe import calculate_limits
    from vlx.config import write_limits

    # Resolve which models to tune
    if all_models:
        models_to_tune = scan_models(MODELS_DIR)
    elif model:
        m = _resolve_model(model)
        models_to_tune = [m]
    else:
        all_m = scan_models(MODELS_DIR)
        if not all_m:
            console.print("[red]No models found.[/red] Run: vlx download")
            raise typer.Exit(1)
        for i, m in enumerate(all_m):
            console.print(f"  {i + 1}) {m.full_name}")
        console.print(f"  {len(all_m) + 1}) All models")
        idx = _pick_number("Which model?", len(all_m) + 1) - 1
        if idx == len(all_m):
            models_to_tune = all_m
        else:
            models_to_tune = [all_m[idx]]

    gpu = get_gpu_info()

    console.print(f"\n[bold]vlx tune[/bold]  [dim]{gpu.name} ({gpu.vram_total_gb:.0f} GB, util {gpu_util:.0%})[/dim]\n")

    for m in models_to_tune:
        # Check cached limits (invalidate if gpu_util changed)
        lim_path = limits_path(m.provider, m.model_name)
        if not recalc:
            existing = read_limits(lim_path)
            if existing and existing.get("gpu_memory_utilization") == gpu_util and existing.get("vram_total_gb") == gpu.vram_total_gb:
                console.print(f"[bold]{m.full_name}[/bold]  [dim](cached — use --recalc to refresh)[/dim]")
                _print_limits_table(existing, m)
                continue

        # Check architecture fields
        if m.num_kv_heads is None or m.head_dim is None or m.num_layers is None:
            console.print(f"[bold]{m.full_name}[/bold]")
            console.print("  [red]Missing architecture fields in config.json[/red]")
            console.print(f"  num_kv_heads={m.num_kv_heads}, head_dim={m.head_dim}, num_layers={m.num_layers}")
            continue

        limits_data = calculate_limits(
            model_info=m,
            vram_total_gb=gpu.vram_total_gb,
            gpu_mem_util=gpu_util,
        )

        write_limits(lim_path, limits_data)
        console.print(f"[bold]{m.full_name}[/bold]  [dim]({m.model_size_gb} GB, {m.quant_method or 'none'})[/dim]")
        _print_limits_table(limits_data, m)
        console.print(f"  [green]Saved to {lim_path}[/green]\n")


def _print_limits_table(limits_data: dict, m: "ModelInfo") -> None:
    """Print the context × concurrency limit table."""
    avail = limits_data.get("available_kv_gb")
    if avail is not None:
        console.print(f"  [dim]KV cache: {avail} GB available[/dim]")

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Context", style="bold")
    table.add_column("Auto KV", justify="right")
    table.add_column("FP8 KV", justify="right")

    for ctx_str in sorted(limits_data.get("limits", {}), key=int):
        entry = limits_data["limits"][ctx_str]
        ctx_val = int(ctx_str)
        ctx_label = f"{ctx_val:,}"

        auto_val = entry.get("auto")
        fp8_val = entry.get("fp8")

        auto_str = f"{auto_val} slots" if auto_val else "[dim]OOM[/dim]"
        fp8_str = f"{fp8_val} slots" if fp8_val else "[dim]OOM[/dim]"

        table.add_row(ctx_label, auto_str, fp8_str)

    console.print(table)
    console.print()


def _launch_vllm(cfg_path: "pathlib.Path", label: str) -> None:
    """Stop any running vLLM, start with given config, wait for health."""
    from vlx.config import read_profile_yaml
    from vlx.serve import start_vllm, stop_vllm, is_vllm_running
    import time

    lock = _lock_or_exit("gpu", f"starting vLLM ({label})")
    cfg = read_profile_yaml(cfg_path) or {}

    try:
        if is_vllm_running():
            console.print("[yellow]vLLM is already running.[/yellow]")
            if not typer.confirm("Stop and restart?", default=False):
                lock.release()
                return
            try:
                stop_vllm()
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
            time.sleep(2)

        ctx = cfg.get("max-model-len", "?")
        ctx_d = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        console.print(f"\n[bold]Starting[/bold] with [cyan]{label}[/cyan]")
        console.print(f"  context: {ctx_d}  kv: {cfg.get('kv-cache-dtype', 'auto')}"
                      f"  seqs: {cfg.get('max-num-seqs', '?')}")

        try:
            start_vllm(cfg_path)
        except RuntimeError as e:
            from vlx.config import cfg as _cfg
            svc = _cfg().service_name
            console.print(f"[red]{e}[/red]")
            console.print(f"  Check: sudo journalctl -u {svc} --no-pager -n 20")
            raise typer.Exit(1)

        from urllib.request import urlopen
        port = cfg.get("port", 8888)
        health_url = f"http://localhost:{port}/health"

        from vlx.config import cfg as _cfg
        fi_cache = _cfg().vllm_root / ".cache" / "flashinfer"
        if not fi_cache.is_dir() or not list(fi_cache.glob("*.so")):
            console.print("  [yellow]First run — compiling kernels (~5-10 min). Subsequent starts will be fast.[/yellow]")

        console.print(f"  [dim]Waiting for {health_url} ...[/dim]")
        for i in range(150):  # 300s max
            time.sleep(2)
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
    finally:
        lock.release()


def _pick_number(prompt: str, max_val: int) -> int:
    while True:
        choice = typer.prompt(prompt)
        try:
            n = int(choice)
            if 1 <= n <= max_val:
                return n
        except ValueError:
            pass
        console.print(f"[red]Enter a number 1-{max_val}.[/red]")


def _custom_config(m: ModelInfo) -> "pathlib.Path":
    """Guide user through manual parameter selection, write a temp config."""
    from vlx.config import read_limits, write_profile_yaml
    from vlx.gpu import get_gpu_info, compute_gpu_memory_utilization

    lim = read_limits(limits_path(m.provider, m.model_name))
    if not lim:
        console.print(f"[red]No probe data for {m.model_name}.[/red] Run: vlx tune {m.model_name}")
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
    write_profile_yaml(cfg_path, cfg, comment="vlx start — custom config")
    console.clear()
    return cfg_path


@app.command()
def start(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
):
    """Start serving a model — interactive config picker."""
    if model is None:
        all_models = scan_models(MODELS_DIR)
        if not all_models:
            console.print("[red]No models found.[/red] Run: vlx download")
            raise typer.Exit(1)

        while True:
            console.print("\n[bold]Select a model:[/bold]")
            for i, m in enumerate(all_models, 1):
                has_limits = read_limits(limits_path(m.provider, m.model_name)) is not None
                status = "[green]tuned[/green]" if has_limits else "[dim]run vlx tune first[/dim]"
                console.print(f"  {i}) {m.full_name}  ({m.model_size_gb} GB)  {status}")

            choice = _pick_number("\nModel number", len(all_models))
            m = all_models[choice - 1]
            if read_limits(limits_path(m.provider, m.model_name)) is not None:
                break
            console.print(f"[yellow]Run vlx tune {m.model_name} first.[/yellow]")
    else:
        m = _resolve_model(model)

    cfg_path = _custom_config(m)
    _launch_vllm(cfg_path, m.model_name)


@app.command()
def stop():
    """Stop the vLLM service."""
    from vlx.serve import stop_vllm, is_vllm_running

    lock = _lock_or_exit("gpu", "stopping vLLM")
    try:
        if not is_vllm_running():
            console.print("[dim]vLLM is not running.[/dim]")
            return

        try:
            stop_vllm()
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        console.print("[green]vLLM stopped.[/green]")
    finally:
        lock.release()


@app.command()
def fan(
    mode: str = typer.Argument(None, help="auto | <30-100> | off"),
):
    """GPU fan control — auto curve, fixed speed, or off."""
    from pathlib import Path

    import vlx.fan as _fan
    from vlx.fan import read_state
    from vlx.gpu import get_fan_speed
    import os
    import signal as sig
    import time

    def _sudo_reexec() -> None:
        """Re-execute this command under sudo if not already root."""
        if os.geteuid() != 0:
            import sys
            os.execvp("sudo", ("sudo", *sys.argv))

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
        # Wait for old daemon to release its fan lock (cleanup in finally block)
        from vlx.lock import wait_for_release
        if not wait_for_release("fan", timeout=5.0):
            console.print("[red]Old fan daemon did not release lock in time.[/red]")
            raise typer.Exit(1)
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

    def _start_fixed_daemon(speed: int) -> None:
        _stop_daemon()
        from vlx.lock import wait_for_release
        if not wait_for_release("fan", timeout=5.0):
            console.print("[red]Old fan daemon did not release lock in time.[/red]")
            raise typer.Exit(1)
        import subprocess
        import sys
        proc = subprocess.Popen(
            [sys.executable, "-m", "vlx.fan", "--fixed", str(speed)],
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
        console.print(f"[green]Fan held at {speed}%[/green] (pid {proc.pid})")

    def _is_orphaned_fan() -> bool:
        """Check if fan daemon crashed, leaving stale PID/state files."""
        if _daemon_pid() is not None:
            return False  # daemon is alive
        _fan._resolve_paths()
        return _fan.PID_PATH.exists() or _fan.STATE_PATH.exists()

    def _clean_orphaned_fan() -> None:
        """Remove stale PID/state files from crashed daemon."""
        _fan._resolve_paths()
        _fan.PID_PATH.unlink(missing_ok=True)
        _fan.STATE_PATH.unlink(missing_ok=True)
        from vlx.gpu import restore_fan_auto
        try:
            restore_fan_auto()
            console.print("[green]Fan auto control restored.[/green]")
        except Exception:
            console.print("[yellow]Could not restore auto fan control via NVML.[/yellow]")

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

        if _is_orphaned_fan():
            console.print("  [red]Warning: fan daemon crashed — fan may be stuck in manual mode[/red]")
            console.print("  [red]Run: sudo vlx fan off[/red]")
        elif state and "fixed" in state:
            console.print(f"  [green]Fixed at {state['fixed']}%[/green] (daemon holding)")
        elif state:
            qs, qe, qm = state["quiet_start"], state["quiet_end"], state["quiet_max"]
            console.print(f"  [green]Auto curve[/green] — quiet {qs:02d}:00-{qe:02d}:00 (max {qm}%), otherwise 100%")
        elif pid:
            console.print("  [green]Daemon running[/green]")
        else:
            console.print("  [dim]No daemon — NVIDIA auto[/dim]")

    # --- Direct mode (non-interactive) ---
    if mode == "auto":
        _sudo_reexec()
        _start_daemon(9, 18, 60)
        return
    if mode == "off":
        if _daemon_pid() or _is_orphaned_fan():
            _sudo_reexec()
        if _stop_daemon():
            console.print("[green]Fan daemon stopped[/green], auto control restored.")
        elif _is_orphaned_fan():
            _clean_orphaned_fan()
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
        _sudo_reexec()
        _start_fixed_daemon(percent)
        return

    # --- Interactive mode ---
    _sudo_reexec()
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
        _start_fixed_daemon(val)
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


@app.command()
def status():
    """Show current vLLM serving status."""
    from vlx.config import active_yaml_path, read_profile_yaml
    from vlx.serve import is_vllm_running

    if not is_vllm_running():
        console.print("[dim]vLLM is not running.[/dim] Start with: vlx start")
        return

    active = active_yaml_path()
    if not (active.exists() and active.is_symlink()):
        console.print("[bold green]vLLM is running[/bold green] (no active config link found)")
        return

    target = active.resolve()
    cfg = read_profile_yaml(active) or {}
    model_path = cfg.get("model", "?")
    model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path

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
    console.print(f"  [bold]Model[/bold]      {model_name}")
    console.print(f"  [bold]Endpoint[/bold]   http://localhost:{port}/v1")
    console.print()

    console.print("  [bold]Serving config[/bold]")
    console.print(f"    Context window:    {ctx_display}")
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
    """Set up vlx — scan the system, write config, optionally install login banner."""
    import shutil
    import subprocess
    from pathlib import Path as _Path

    from vlx.config import (
        CONFIG_FILE, save_config,
        _discover_vllm_root, _discover_cuda_home, _discover_service, _discover_port,
        _build_config, reset_config,
    )

    def _ok(msg: str) -> None:
        console.print(f"  [green]{msg}[/green]")

    def _warn(msg: str) -> None:
        console.print(f"  [yellow]{msg}[/yellow]")

    def _fail(msg: str) -> None:
        console.print(f"  [red]{msg}[/red]")

    console.print("\n[bold]vlx setup[/bold]\n")
    console.print("[dim]Scanning your system...[/dim]\n")

    # ── GPU ──
    try:
        from vlx.gpu import get_gpu_info
        gpu = get_gpu_info()
        _ok(f"GPU         {gpu.name} ({gpu.vram_total_gb:.0f} GB)")
        _ok(f"Driver      {gpu.driver}")
        _ok(f"CUDA        {gpu.cuda}")
    except Exception:
        _fail("GPU         nvidia-smi not found or no GPU detected")
        _fail("            vlx requires an NVIDIA GPU with drivers installed")
        _fail("            Install drivers: https://www.nvidia.com/drivers")

    # ── CUDA toolkit ──
    cuda = _discover_cuda_home()
    nvcc = shutil.which("nvcc")
    if nvcc:
        _ok(f"nvcc        {cuda}")
    else:
        _fail("nvcc        not on PATH — needed for first-run JIT compilation")
        _fail("            Install: sudo apt install nvidia-cuda-toolkit")
        _warn(f"            using fallback: {cuda}")

    # ── vLLM ──
    root = _discover_vllm_root()
    if root:
        # Check vLLM version
        vllm_bin = root / "venv" / "bin" / "vllm"
        if not vllm_bin.exists():
            vllm_bin = root / ".venv" / "bin" / "vllm"
        try:
            r = subprocess.run(
                [str(vllm_bin), "--version"],
                capture_output=True, text=True, timeout=10,
            )
            ver = r.stdout.strip() or r.stderr.strip()
            _ok(f"vLLM        {ver} ({root})")
        except Exception:
            _ok(f"vLLM        found at {root}")
    else:
        _fail("vLLM        not found. Install it first: pip install vllm")
        _fail("            See: https://docs.vllm.ai/en/latest/getting_started/installation.html")
        root_input = typer.prompt("  vLLM root path", default="/opt/vllm")
        root = _Path(root_input)

    # ── Models ──
    models_dir = root / "models"
    if models_dir.is_dir():
        models = list(models_dir.glob("*/*/config.json"))
        _ok(f"Models      {len(models)} found at {models_dir}")
    else:
        _warn(f"Models      {models_dir} does not exist")

    # ── systemd ──
    svc_name, svc_user = _discover_service()
    svc_path = _Path(f"/etc/systemd/system/{svc_name}.service")
    if svc_path.exists():
        _ok(f"systemd     {svc_name}.service (user: {svc_user})")
    else:
        _fail("systemd     no vLLM systemd service found (required for vlx start/stop)")
        _fail(f"            Create /etc/systemd/system/{svc_name}.service with:")
        _fail("            [Service] ExecStart=/opt/vllm/venv/bin/vllm serve ...")

    # ── Port ──
    port = _discover_port(root)
    _ok(f"Port        {port}")

    # ── Coolbits (fan control) ──
    xorg_conf = _Path("/etc/X11/xorg.conf")
    if xorg_conf.exists() and "Coolbits" in xorg_conf.read_text():
        _ok("Coolbits    enabled (fan control available)")
    else:
        _warn("Coolbits    not configured (vlx fan requires it)")

    # ── gum (interactive UI) ──
    if shutil.which("gum"):
        _ok("gum         installed (interactive UI enabled)")
    else:
        _warn("gum         not installed (vlx download will use basic mode)")

    # ── Write config ──
    console.print()
    if CONFIG_FILE.exists():
        console.print(f"  [dim]Config exists at {CONFIG_FILE}[/dim]")
        if not typer.confirm("  Overwrite config?", default=True):
            console.print()
        else:
            config = _build_config(root, cuda, svc_name, svc_user, port)
            save_config(config)
            reset_config()
            _ok(f"Config written to {CONFIG_FILE}")
    else:
        config = _build_config(root, cuda, svc_name, svc_user, port)
        save_config(config)
        reset_config()
        _ok(f"Config written to {CONFIG_FILE}")

    # ── Welcome banner ──
    banner_src = _Path(__file__).parent / "welcome.sh"
    banner_dest = CONFIG_FILE.parent / "welcome.sh"
    _marker = "# vlx login banner"

    # Detect shell rc file
    import os
    _shell = _Path(os.environ.get("SHELL", "/bin/bash")).name
    if _shell == "zsh":
        _rc = _Path.home() / ".zshrc"
    elif _shell == "fish":
        _rc = _Path.home() / ".config" / "fish" / "config.fish"
    else:
        _rc = _Path.home() / ".bashrc"

    _source_line = f'[ -f "{banner_dest}" ] && source "{banner_dest}"'
    _rc_has_marker = _rc.exists() and _marker in _rc.read_text()
    has_banner = banner_dest.exists() and _rc_has_marker

    if _shell == "fish":
        console.print("  [yellow]Login banner not supported for fish shell.[/yellow]")
    elif has_banner:
        console.print(f"  [dim]Login banner already installed ({_rc.name} → {banner_dest})[/dim]")
        if typer.confirm("  Reinstall login banner?", default=False):
            import shutil
            shutil.copy2(banner_src, banner_dest)
            _ok(f"Banner updated at {banner_dest}")
    else:
        console.print()
        console.print("  [bold]Login banner[/bold]")
        console.print("  Shows GPU status, model, and commands on every SSH login.")
        console.print("  Requires: gum")
        if typer.confirm("  Install login banner?", default=True):
            import shutil
            shutil.copy2(banner_src, banner_dest)
            if not _rc_has_marker:
                with open(_rc, "a") as f:
                    f.write(f"\n{_marker}\n{_source_line}\n")
            _ok(f"Banner installed ({_rc.name} → {banner_dest})")
        else:
            console.print("  [dim]Skipped.[/dim]")

    # ── Next steps ──
    console.print()
    console.print("[bold]Next steps[/bold]")
    console.print("  vlx download    Search & download a model")
    console.print("  vlx doctor      Full system check")
    console.print("  vlx             Dashboard")
    console.print()


@app.command()
def doctor():
    """Check system readiness for vLLM serving."""
    import os
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

    console.print("\n[bold]vlx doctor[/bold]\n")

    # -- Environment --
    console.print("  [bold]Environment[/bold]")

    # nvcc
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5,
                           env={**os.environ, "PATH": "/usr/local/cuda/bin:" + os.environ.get("PATH", "")})
        if r.returncode == 0:
            ver = [ln for ln in r.stdout.splitlines() if "release" in ln]
            _ok(f"nvcc {ver[0].split('release')[-1].strip().rstrip(',') if ver else 'found'}")
        else:
            _fail("nvcc not working", "Install CUDA toolkit or check /usr/local/cuda/bin/nvcc")
    except Exception:
        _fail("nvcc not found", "Install: sudo apt install nvidia-cuda-toolkit  OR  https://developer.nvidia.com/cuda-downloads")

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
        _fail("GPU not accessible", "Install NVIDIA drivers: https://www.nvidia.com/drivers  then check: nvidia-smi")

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
        _fail(f"No systemd unit at {svc_path}",
              "Create one: https://docs.vllm.ai  or see vlx docs/troubleshooting.md")

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
            _warn(f"{len(stale)} stale sockets in {tmp_dir}", "vlx cache clean")
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
        _warn("No active.yaml — run vlx start to configure")

    # Port
    _port = _c.port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        port_in_use = s.connect_ex(("127.0.0.1", _port)) == 0
    if is_vllm_running():
        _ok(f"vLLM serving on port {_port}")
    elif port_in_use:
        _fail(f"Port {_port} in use but vLLM not running — something else is bound",
              f"Check: sudo lsof -i :{_port}  then stop the other service or change port in ~/.config/vx/config.yaml")
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

    # Multi-user messaging
    import grp
    try:
        tty_members = grp.getgrnam("tty").gr_mem
        import os
        me = os.environ.get("USER", "")
        if me in tty_members:
            _ok("tty group (terminal messaging between users)")
        else:
            _warn(f"'{me}' not in tty group — vlx can't DM other users",
                  f"sudo usermod -aG tty {me}  (then re-login)")
    except KeyError:
        _warn("tty group not found")

    # Models
    all_models = scan_models(MODELS_DIR)
    probed = sum(1 for m in all_models if read_limits(limits_path(m.provider, m.model_name)))
    _ok(f"{len(all_models)} models downloaded, {probed} probed")

    console.print(f"\n  [bold]{'All clear' if fail_count == 0 else f'{fail_count} issue(s) found'}[/bold]  "
                  f"({ok_count} ok, {fail_count} fail)\n")


