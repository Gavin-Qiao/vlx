"""vserve — LLM inference manager CLI."""


import pathlib
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vserve.config import (
    CONFIG_FILE,
    MODELS_DIR,
    limits_path,
    profile_path,
    read_limits,
)
from vserve.models import scan_models, fuzzy_match, ModelInfo
from vserve.lock import VserveLock, LockHeld, SessionHeld, notify_user, check_session, write_session, clear_session

from vserve import __version__

app = typer.Typer(help="LLM inference manager")
cache_app = typer.Typer(help="Cache management")
app.add_typer(cache_app, name="cache")
console = Console()
_TITLE = f"[bold cyan]vserve[/bold cyan] [dim]{__version__}[/dim]"



def _session_or_exit() -> None:
    """Block if another user owns the active GPU session, DM the holder."""
    import os
    try:
        check_session()
    except SessionHeld as exc:
        console.print(Panel(
            f"[bold]{exc.message()}[/bold]"
            f"\n\nAsk [cyan]{exc.info.user}[/cyan] to run [cyan]vserve stop[/cyan] first.",
            title=f"[red]vserve {__version__}: session locked[/red]",
            border_style="red",
        ))
        me = os.environ.get("USER", "?")
        if exc.info.user != me:
            notify_user(
                exc.info.user,
                f"{me} wants the GPU (you: {exc.info.model})",
            )
        raise typer.Exit(1) from None


def _lock_or_exit(name: str, description: str) -> VserveLock:
    """Acquire a lock or DM the holder and exit."""
    import os
    lock = VserveLock(name, description)
    try:
        lock.acquire()
    except LockHeld as exc:
        console.print(Panel(
            f"[bold]{exc.message()}[/bold]",
            title=f"[red]vserve {__version__}: {name} locked[/red]",
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
    except PermissionError as exc:
        console.print(Panel(
            f"[bold]{exc}[/bold]",
            title=f"[red]vserve {__version__}: permission denied[/red]",
            border_style="red",
        ))
        raise typer.Exit(1) from None
    return lock


def _all_models() -> list[ModelInfo]:
    """Scan model directories for all registered backends."""
    from vserve.backends import _BACKENDS
    all_m: list[ModelInfo] = []
    seen: set = set()
    # Always include the legacy MODELS_DIR (vLLM)
    for m in scan_models(MODELS_DIR):
        if m.path not in seen:
            all_m.append(m)
            seen.add(m.path)
    # Scan additional backend model dirs
    for b in _BACKENDS:
        models_dir = b.root_dir / "models"
        if models_dir == MODELS_DIR or not models_dir.exists():
            continue
        for m in scan_models(models_dir):
            if m.path not in seen:
                all_m.append(m)
                seen.add(m.path)
    return all_m


def _resolve_model(query: str) -> ModelInfo:
    models = _all_models()
    if not models:
        console.print("[red]No models found.[/red] Run: vserve download")
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


def _show_update_notice() -> None:
    """Print Wrangler-style update box if a newer version is available."""
    from vserve.version import update_available
    info = update_available()
    if info:
        console.print(Panel(
            f"  vserve [bold]{info.current}[/bold] → [bold green]{info.latest}[/bold green]\n"
            f"  Run [cyan]vserve update[/cyan] to upgrade",
            title=f"[yellow]vserve {__version__}: update available[/yellow]",
            border_style="yellow",
        ))


@app.callback(invoke_without_command=True)
def dashboard(ctx: typer.Context):
    """Show status dashboard when called with no subcommand."""
    if ctx.invoked_subcommand is not None:
        from vserve.version import background_refresh
        background_refresh()
        from rich.rule import Rule
        console.print(Rule(title=f"[bold cyan]vserve[/bold cyan] [dim]{__version__}[/dim]", style="cyan"))
        if ctx.invoked_subcommand not in ("version", "update"):
            def _close_border() -> None:
                _show_update_notice()
                console.print(Rule(style="cyan"))
            ctx.call_on_close(_close_border)
        else:
            ctx.call_on_close(lambda: console.print(Rule(style="cyan")))
        return

    if not CONFIG_FILE.exists():
        console.print(Panel(
            "[bold]First time? Run [cyan]vserve init[/cyan] to set up your system.[/bold]",
            title=_TITLE,
            border_style="yellow",
        ))

    try:
        from vserve.gpu import get_gpu_info
        gpu = get_gpu_info()
        gpu_line = f"{gpu.name} ({gpu.vram_total_gb:.0f} GB, CUDA {gpu.cuda})"
    except Exception:
        gpu_line = "[dim]unavailable[/dim]"

    from vserve.backends import running_backend as _running_backend
    from vserve.config import active_yaml_path, read_profile_yaml

    models = _all_models()
    probed = sum(
        1 for m in models if read_limits(limits_path(m.provider, m.model_name))
    )

    serving_line = "[dim]not running[/dim]"
    rb = _running_backend()
    if rb is not None:
        active = active_yaml_path()
        if active.exists() and active.is_symlink():
            cfg = read_profile_yaml(active)
            model_path = cfg.get("model", "?") if cfg else "?"
            model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path
            port = cfg.get("port", 8888) if cfg else 8888
            serving_line = f"[green]{model_name}[/green] at :{port} ({rb.display_name})"
        else:
            serving_line = f"[green]active[/green] ({rb.display_name})"

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
        ("stop", "Stop the inference server"),
        ("status", "Show what's currently serving"),
        ("models", "List models with limits & profiles"),
        ("tune [model]", "Calculate context & concurrency limits"),
        ("fan [auto|off|30-100]", "GPU fan control with temp curve"),
        ("doctor", "Check system readiness"),
        ("init", "Auto-discover backends and write config"),
    ]:
        cmd_tbl.add_row(_Txt(cmd, style="bold cyan"), desc)

    from rich.console import Group
    body = Group(
        "\n".join(lines) + "\n  [bold]Commands[/bold]",
        cmd_tbl,
    )
    console.print(Panel(body, title=_TITLE, border_style="cyan"))
    from vserve.version import background_refresh
    background_refresh()
    _show_update_notice()


@app.command()
def version():
    """Show current version and check for updates."""
    from vserve.version import update_available, background_refresh

    background_refresh()
    info = update_available()

    if info:
        console.print(f"  vserve [bold]{info.current}[/bold]")
        console.print(f"  [yellow]Update available: {info.latest}[/yellow]")
        console.print("  Run [cyan]vserve update[/cyan] to upgrade")
    else:
        console.print(f"  vserve [bold]{__version__}[/bold] — up to date")


@app.command()
def update():
    """Update vserve to the latest version."""
    import shutil
    import subprocess

    from vserve.version import check_pypi, write_cache, _compare_versions

    console.print(f"[dim]Current version: {__version__}[/dim]")

    latest = check_pypi()
    if latest:
        write_cache(latest)
        if _compare_versions(latest, __version__) <= 0:
            console.print("[green]Already up to date.[/green]")
            return
        console.print(f"[yellow]New version available: {latest}[/yellow]\n")

    uv = shutil.which("uv")
    if uv:
        result = subprocess.run([uv, "tool", "list"], capture_output=True, text=True)
        if "vserve" in result.stdout:
            console.print("[dim]Upgrading via uv...[/dim]")
            subprocess.run([uv, "tool", "upgrade", "vserve"])
            return

    pip = shutil.which("pip") or shutil.which("pip3")
    if pip:
        console.print("[dim]Upgrading via pip...[/dim]")
        subprocess.run([pip, "install", "--upgrade", "vserve"])
        return

    console.print("[red]Could not find uv or pip to perform upgrade.[/red]")
    console.print("Run manually: [cyan]uv tool upgrade vserve[/cyan] or [cyan]pip install -U vserve[/cyan]")


@app.command()
def models(model: str = typer.Argument(None, help="Model name for detail view")):
    """List downloaded models."""
    from vserve.backends import _BACKENDS

    all_models = _all_models()
    if not all_models:
        console.print("[dim]No models found.[/dim] Run: vserve download")
        return

    if model:
        m = _resolve_model(model)
        _show_model_detail(m)
        return

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="bold")
    table.add_column("Backend", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Limits")
    table.add_column("Max Context", justify="right")
    table.add_column("Tools", style="green")
    table.add_column("Reasoning", style="green")

    for m in all_models:
        lim = read_limits(limits_path(m.provider, m.model_name))

        # Detect backend
        backend_name = "\u2014"
        for b in _BACKENDS:
            if b.can_serve(m):
                backend_name = b.display_name
                break

        max_ctx = "\u2014"
        if lim:
            for ctx_str in sorted(lim.get("limits", {}).keys(), key=lambda x: int(x), reverse=True):
                entry = lim["limits"][ctx_str]
                if isinstance(entry, dict):
                    if any(v is not None for v in entry.values()):
                        max_ctx = f"{int(ctx_str) // 1024}k"
                        break
                elif entry is not None:
                    max_ctx = f"{int(ctx_str) // 1024}k"
                    break

        # Detect tools via backend
        tool_info = {}
        for b in _BACKENDS:
            if b.can_serve(m):
                tool_info = b.detect_tools(m.path)
                break
        tp = tool_info.get("tool_call_parser") or ("\u2713" if tool_info.get("supports_tools") else "\u2014")
        rp = tool_info.get("reasoning_parser", "\u2014") or "\u2014"

        table.add_row(
            m.full_name, backend_name, f"{m.model_size_gb} GB",
            "\u2713" if lim else "\u2717", max_ctx, tp, rp,
        )

    console.print(table)


def _show_model_detail(m: ModelInfo):
    lim = read_limits(limits_path(m.provider, m.model_name))

    console.print(f"\n[bold]{m.full_name}[/bold]")
    console.print(f"  Arch: {m.architecture}  Quant: {m.quant_method or 'none'}  MoE: {m.is_moe}")
    console.print(f"  Size: {m.model_size_gb} GB  Max positions: {m.max_position_embeddings}\n")

    if not lim:
        console.print(f"  [dim]Not probed yet.[/dim] Run: vserve tune {m.model_name.lower()}\n")
        return

    limits = lim.get("limits", {})
    is_flat = any(isinstance(v, (int, type(None))) for v in limits.values())

    if is_flat:
        table = Table(title="Context / Concurrency Limits")
        table.add_column("Context", justify="right")
        table.add_column("Parallel slots", justify="right")
        for ctx_str in sorted(limits.keys(), key=int):
            entry = limits[ctx_str]
            slot_str = str(entry) if entry is not None else "OOM"
            table.add_row(f"{int(ctx_str) // 1024}k", slot_str)
    else:
        table = Table(title="Context / Concurrency Limits")
        table.add_column("Context", justify="right")
        table.add_column("kv auto", justify="right")
        table.add_column("kv fp8", justify="right")
        for ctx_str in sorted(limits.keys(), key=int):
            entry = limits[ctx_str]
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
    from vserve.config import cfg as _cfg

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
        from vserve.variants import format_variant_line
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
    from vserve.variants import fetch_repo_variants, format_variant_line, _format_bytes

    parts = model_id.split("/")
    if len(parts) != 2:
        console.print(f"[red]Invalid model ID '{model_id}'.[/red] Expected: provider/model-name")
        raise typer.Exit(1)

    provider, model_name = parts

    # Determine destination — will be updated after variant selection for GGUF
    local_dir = models_dir / provider / model_name
    is_gguf_download = False

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

    # Check if this is a GGUF download — route to llama-cpp models dir
    is_gguf_download = any(f.endswith(".gguf") for f in allow_files)
    if is_gguf_download:
        from vserve.config import cfg as _cfg2
        lc_root = _cfg2().llamacpp_root
        if lc_root:
            local_dir = lc_root / "models" / provider / model_name
        else:
            console.print("[yellow]llama.cpp not configured — downloading GGUF to default models dir[/yellow]")

    console.print(f"\n[bold]Downloading[/bold] {model_id}")
    console.print(f"  To: {local_dir}\n")

    lock_name = f"download-{provider}--{model_name}"
    lock = VserveLock(lock_name, f"downloading {model_id}")
    try:
        lock.acquire()
    except PermissionError as exc:
        console.print(Panel(
            f"[bold]{exc}[/bold]",
            title=f"[red]vserve {__version__}: permission denied[/red]",
            border_style="red",
        ))
        raise typer.Exit(1) from None
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
        from vserve.lock import wait_for_release
        if not wait_for_release(lock_name, timeout=7200):
            console.print("[red]Timed out waiting for download.[/red]")
            raise typer.Exit(1)
        if local_dir.exists() and any(local_dir.iterdir()):
            try:
                from vserve.models import detect_model
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
        except PermissionError as exc:
            console.print(Panel(
                f"[bold]{exc}[/bold]",
                title=f"[red]vserve {__version__}: permission denied[/red]",
                border_style="red",
            ))
            raise typer.Exit(1) from None

    try:
        if is_gguf_download:
            from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
            gguf_files = [f for f in allow_files if f.endswith(".gguf")]
            other_files = [f for f in allow_files if not f.endswith(".gguf")]
            # Download GGUF files individually
            for filename in gguf_files:
                hf_hub_download(repo_id=model_id, filename=filename, local_dir=local_dir)
            # Also grab tokenizer_config.json for tool detection
            try:
                hf_hub_download(repo_id=model_id, filename="tokenizer_config.json", local_dir=local_dir)
            except Exception:
                pass  # not all GGUF repos have it
            # Download any shared non-GGUF files
            for filename in other_files:
                try:
                    hf_hub_download(repo_id=model_id, filename=filename, local_dir=local_dir)
                except Exception:
                    pass
        else:
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
        from vserve.models import detect_model
        info = detect_model(local_dir)
    except Exception:
        info = None
    if info:
        console.print(f"\n[green]Downloaded {info.full_name}[/green]")
        console.print(f"  Size: {info.model_size_gb:.1f} GB")
        console.print(f"  Quant: {info.quant_method or 'none'}")
        console.print(f"  Context: {info.max_position_embeddings:,} tokens")
        console.print(f"\nNext: vserve tune {model_name}")
    else:
        console.print(f"\n[green]Downloaded to {local_dir}[/green]")
    return True


@app.command()
def tune(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    all_models: bool = typer.Option(False, "--all", help="Tune all downloaded models"),
    recalc: bool = typer.Option(False, "--recalc", help="Force recalculation even if cached"),
    gpu_util: float = typer.Option(None, "--gpu-util", help="GPU memory utilization (0.5-0.99, auto if omitted)"),
):
    """Calculate context and concurrency limits for a model."""
    if gpu_util is not None and not 0.5 <= gpu_util <= 0.99:
        console.print("[red]--gpu-util must be between 0.5 and 0.99[/red]")
        raise typer.Exit(1)

    from vserve.gpu import get_gpu_info, compute_gpu_memory_utilization
    from vserve.config import write_limits

    # Resolve which models to tune
    if all_models:
        models_to_tune = _all_models()
    elif model:
        m = _resolve_model(model)
        models_to_tune = [m]
    else:
        all_m = _all_models()
        if not all_m:
            console.print("[red]No models found.[/red] Run: vserve download")
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
    if gpu_util is None:
        from vserve.config import cfg as _cfg, save_config, reset_config
        _c = _cfg()
        if _c.gpu_memory_utilization is not None:
            gpu_util = _c.gpu_memory_utilization
        elif _c.gpu_overhead_gb is not None:
            gpu_util = compute_gpu_memory_utilization(gpu.vram_total_gb, _c.gpu_overhead_gb)
        else:
            gpu_util = compute_gpu_memory_utilization(gpu.vram_total_gb)
            # Interactive menu only when user runs bare `vserve tune` (no model arg)
            if model is None and not all_models:
                console.print(f"\n[bold]GPU memory reservation[/bold]  [dim]({gpu.vram_total_gb:.0f} GB total, current: {gpu_util:.0%})[/dim]")
                console.print("  1) Auto (3.5 GB overhead) — use for this run")
                console.print("  2) Set GPU usage % (save to config)")
                console.print("  3) Set overhead in GB (save to config)")
                choice = typer.prompt("\n  Choice", default="1")
                if choice == "2":
                    val = typer.prompt("  GPU usage %", default=f"{gpu_util:.2f}")
                    gpu_util = max(0.5, min(0.99, float(val)))
                    _c.gpu_memory_utilization = gpu_util
                    _c.gpu_overhead_gb = None
                    save_config(_c)
                    reset_config()
                    console.print(f"  [green]Saved gpu_memory_utilization={gpu_util:.2%}[/green]")
                elif choice == "3":
                    val = typer.prompt("  Overhead GB", default="3.5")
                    overhead = max(0.5, float(val))
                    gpu_util = compute_gpu_memory_utilization(gpu.vram_total_gb, overhead)
                    _c.gpu_overhead_gb = overhead
                    _c.gpu_memory_utilization = None
                    save_config(_c)
                    reset_config()
                    console.print(f"  [green]Saved gpu_overhead_gb={overhead} GB[/green]")

    console.print(f"\n[bold]vserve tune[/bold]  [dim]{gpu.name} ({gpu.vram_total_gb:.0f} GB, util {gpu_util:.0%})[/dim]\n")

    for m in models_to_tune:
        from vserve.backends import get_backend
        try:
            backend = get_backend(m)
        except ValueError:
            console.print(f"[bold]{m.full_name}[/bold]  [red]No backend available[/red]")
            continue

        # Check cached limits (invalidate if gpu_util changed)
        lim_path = limits_path(m.provider, m.model_name)
        if not recalc:
            existing = read_limits(lim_path)
            if existing and existing.get("gpu_memory_utilization") == gpu_util and existing.get("vram_total_gb") == gpu.vram_total_gb:
                console.print(f"[bold]{m.full_name}[/bold]  [dim](cached — use --recalc to refresh)[/dim]")
                _print_limits_table(existing, m)
                continue

        # For vLLM backend, check architecture fields
        if backend.name == "vllm" and (m.num_kv_heads is None or m.head_dim is None or m.num_layers is None):
            console.print(f"[bold]{m.full_name}[/bold]")
            console.print("  [red]Missing architecture fields in config.json[/red]")
            console.print(f"  num_kv_heads={m.num_kv_heads}, head_dim={m.head_dim}, num_layers={m.num_layers}")
            continue

        try:
            limits_data = backend.tune(m, gpu, gpu_mem_util=gpu_util)
        except Exception as e:
            console.print(f"[bold]{m.full_name}[/bold]  [red]{e}[/red]")
            continue

        write_limits(lim_path, limits_data)
        console.print(f"[bold]{m.full_name}[/bold]  [dim]({m.model_size_gb} GB, {backend.display_name})[/dim]")
        _print_limits_table(limits_data, m)
        tp = limits_data.get("tool_call_parser")
        rp = limits_data.get("reasoning_parser")
        supports = limits_data.get("supports_tools")
        if tp or rp:
            parts = []
            if tp:
                parts.append(f"tools=[green]{tp}[/green]")
            if rp:
                parts.append(f"reasoning=[green]{rp}[/green]")
            console.print(f"  Capabilities: {' '.join(parts)}")
        elif supports:
            # llama.cpp: tools supported via --jinja, no parser name needed
            console.print("  Capabilities: [green]tool calling (--jinja)[/green]")
        else:
            from vserve.tools import supports_tools as _supports_tools
            if _supports_tools(m.path):
                console.print("  Capabilities: [yellow]tool markers found but parser unknown[/yellow]")
                console.print("                use --tools --tool-parser <parser> with vserve start")
            else:
                console.print("  Capabilities: [dim]no tool calling detected[/dim]")
        console.print(f"  [green]Saved to {lim_path}[/green]\n")


def _print_limits_table(limits_data: dict, m: "ModelInfo") -> None:
    """Print the context × concurrency limit table."""
    avail = limits_data.get("available_kv_gb")
    if avail is not None:
        console.print(f"  [dim]KV cache: {avail} GB available[/dim]")

    # Detect format: llama.cpp uses flat int values, vLLM uses nested dicts
    limits = limits_data.get("limits", {})
    is_flat = any(isinstance(v, (int, type(None))) for v in limits.values())

    if is_flat:
        # llama.cpp format: {"4096": 8, "8192": 4, ...}
        n_layers = limits_data.get("n_gpu_layers")
        num_layers = limits_data.get("num_layers")
        if n_layers is not None and num_layers:
            offload = "full" if n_layers >= num_layers else f"{n_layers}/{num_layers} layers"
            console.print(f"  [dim]GPU offload: {offload}[/dim]")

        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Context", style="bold")
        table.add_column("Parallel slots", justify="right")

        for ctx_str in sorted(limits, key=int):
            entry = limits[ctx_str]
            ctx_val = int(ctx_str)
            ctx_label = f"{ctx_val:,}"
            slot_str = f"{entry} slots" if entry else "[dim]OOM[/dim]"
            table.add_row(ctx_label, slot_str)

        console.print(table)
    else:
        # vLLM format: {"4096": {"auto": 64, "fp8": 128}, ...}
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Context", style="bold")
        table.add_column("Auto KV", justify="right")
        table.add_column("FP8 KV", justify="right")

        for ctx_str in sorted(limits, key=int):
            entry = limits[ctx_str]
            ctx_val = int(ctx_str)
            ctx_label = f"{ctx_val:,}"

            auto_val = entry.get("auto")
            fp8_val = entry.get("fp8")

            auto_str = f"{auto_val} slots" if auto_val else "[dim]OOM[/dim]"
            fp8_str = f"{fp8_val} slots" if fp8_val else "[dim]OOM[/dim]"

            table.add_row(ctx_label, auto_str, fp8_str)

        console.print(table)
    console.print()


def _launch_backend(backend, cfg_path: "pathlib.Path", label: str) -> None:
    """Stop any running backend, start with given config, wait for health."""
    import json
    import time
    from vserve.config import read_profile_yaml

    lock = _lock_or_exit("gpu", f"starting {backend.display_name} ({label})")

    # Read config (YAML for vLLM, JSON for llama.cpp)
    if str(cfg_path).endswith(".json"):
        cfg = json.loads(cfg_path.read_text())
    else:
        cfg = read_profile_yaml(cfg_path) or {}

    try:
        _session_or_exit()

        if backend.is_running():
            console.print(f"[yellow]{backend.display_name} is already running.[/yellow]")
            if not typer.confirm("Stop and restart?", default=False):
                lock.release()
                return
            clear_session()
            try:
                backend.stop()
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
            time.sleep(2)

        ctx = cfg.get("max-model-len") or cfg.get("ctx-size", "?")
        ctx_d = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        console.print(f"\n[bold]Starting[/bold] with [cyan]{label}[/cyan] ({backend.display_name})")
        console.print(f"  context: {ctx_d}")

        try:
            backend.start(cfg_path)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 20")
            raise typer.Exit(1)

        from urllib.request import urlopen
        port = cfg.get("port", 8888)
        health = backend.health_url(port)

        # Backend-specific first-run messages
        if backend.name == "vllm":
            from vserve.config import cfg as _cfg
            fi_cache = _cfg().vllm_root / ".cache" / "flashinfer"
            if not fi_cache.is_dir() or not list(fi_cache.glob("**/*.so")):
                console.print("  [yellow]First run — compiling kernels (~5-10 min).[/yellow]")

        console.print(f"  [dim]Waiting for {health} ...[/dim]")
        for i in range(150):  # 300s max
            time.sleep(2)
            try:
                resp = urlopen(health, timeout=2)
                if resp.status == 200:
                    write_session(label)
                    console.print(f"\n[bold green]{backend.display_name} is running[/bold green] at http://localhost:{port}/v1")
                    console.print(f"  Config: {cfg_path}")
                    console.print(f"  Logs:   sudo journalctl -u {backend.service_name} -f\n")
                    return
            except Exception:
                pass
            if i > 5 and not backend.is_running():
                console.print(f"[red]Service stopped unexpectedly.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
                raise typer.Exit(1)
            if i > 0 and i % 15 == 0:
                console.print(f"  [dim]still starting... ({i * 2}s)[/dim]")
        console.print(f"[red]Timed out waiting for health endpoint.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
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


def _custom_config(m: ModelInfo, backend, *, tools: bool = False, tool_parser: str | None = None) -> "pathlib.Path":
    """Guide user through parameter selection, delegate config building to backend."""
    if backend.name == "llamacpp":
        return _custom_config_llamacpp(m, backend, tools=tools)
    return _custom_config_vllm(m, backend, tools=tools, tool_parser=tool_parser)


def _custom_config_llamacpp(m: ModelInfo, backend, *, tools: bool = False) -> "pathlib.Path":
    """Interactive config wizard for llama.cpp models."""
    import json
    from vserve.config import read_limits, write_limits as _write_limits
    from vserve.gpu import get_gpu_info, compute_gpu_memory_utilization

    gpu = get_gpu_info()
    gpu_mem_util = compute_gpu_memory_utilization(gpu.vram_total_gb)

    lim_path_ = limits_path(m.provider, m.model_name)
    lim = read_limits(lim_path_)
    if not lim:
        console.print(f"[dim]  Auto-tuning {m.model_name}...[/dim]")
        lim = backend.tune(m, gpu, gpu_mem_util=gpu_mem_util)
        _write_limits(lim_path_, lim)
        console.print("[green]  Tuned.[/green]")

    n_gpu_layers = lim.get("n_gpu_layers", 0)
    num_layers = lim.get("num_layers", 0)
    full_offload = lim.get("full_offload", True)
    limits = lim.get("limits", {})

    # 1. Context window
    working_ctxs = sorted(int(c) for c, v in limits.items() if v is not None)
    if not working_ctxs:
        console.print("[red]No working configs found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Configure {m.model_name}[/bold] (llama.cpp)\n")
    console.print("  [bold]1. Context window[/bold]")
    for i, ctx in enumerate(working_ctxs, 1):
        slots = limits.get(str(ctx))
        slot_str = f"({slots} slots)" if slots else ""
        console.print(f"     {i}) {ctx // 1024}k  {slot_str}")
    ctx_idx = _pick_number("\n  Context", len(working_ctxs))
    chosen_ctx = working_ctxs[ctx_idx - 1]

    # 2. GPU layers
    if not full_offload:
        console.print(f"\n  [bold]2. GPU layers[/bold] (max: {num_layers}, fits: {n_gpu_layers})")
        console.print(f"     [yellow]Model partially fits — {n_gpu_layers}/{num_layers} layers on GPU[/yellow]")
        layers_str = typer.prompt(f"  Layers [1-{num_layers}]", default=str(n_gpu_layers))
        try:
            n_gpu_layers = min(max(1, int(layers_str)), num_layers)
        except ValueError:
            pass
    else:
        console.print(f"\n  [dim]GPU layers: {n_gpu_layers}/{num_layers} (all on GPU)[/dim]")

    # 3. Parallel slots
    max_parallel = limits.get(str(chosen_ctx), 1) or 1
    console.print(f"\n  [bold]3. Parallel slots[/bold] (max: {max_parallel})")
    par_str = typer.prompt(f"  Slots [1-{max_parallel}]", default=str(max_parallel))
    try:
        chosen_parallel = min(max(1, int(par_str)), max_parallel)
    except ValueError:
        chosen_parallel = max_parallel

    # 4. Tool calling
    tool_info = backend.detect_tools(m.path)
    supports = tool_info.get("supports_tools", False)
    if supports:
        console.print("\n  [bold]4. Tool calling[/bold]")
        if not tools:
            tools = typer.confirm("     Enable tool calling? (--jinja)", default=False)
        else:
            console.print("     [green]Enabled[/green] (--jinja)")

    # Build config via backend
    choices = {
        "context": chosen_ctx,
        "n_gpu_layers": n_gpu_layers,
        "parallel": chosen_parallel,
        "port": 8888,
        "tools": tools,
    }
    cfg = backend.build_config(m, choices)

    # Summary
    console.print("\n  [bold]Summary[/bold]")
    console.print(f"    Context:     {chosen_ctx // 1024}k")
    console.print(f"    GPU layers:  {n_gpu_layers}/{num_layers}")
    console.print(f"    Parallel:    {chosen_parallel}")
    if tools:
        console.print("    Tool calling: [green]enabled (--jinja)[/green]")

    confirm = typer.prompt("\n  Start? [Y/n]", default="Y")
    if confirm.strip().lower() == "n":
        raise typer.Exit(0)

    # Write JSON config for llama-server
    cfg_dir = backend.root_dir / "configs" / "models"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{m.provider}--{m.model_name}.custom.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    console.clear()
    return cfg_path


def _custom_config_vllm(m: ModelInfo, backend, *, tools: bool = False, tool_parser: str | None = None) -> "pathlib.Path":
    """Interactive config wizard for vLLM models."""
    from vserve.config import read_limits, write_profile_yaml
    from vserve.gpu import get_gpu_info, compute_gpu_memory_utilization

    gpu = get_gpu_info()
    gpu_mem_util = compute_gpu_memory_utilization(gpu.vram_total_gb)

    lim_path_ = limits_path(m.provider, m.model_name)
    lim = read_limits(lim_path_)
    if not lim:
        console.print(f"[dim]  Auto-tuning {m.model_name}...[/dim]")
        from vserve.config import write_limits as _write_limits
        lim = backend.tune(m, gpu, gpu_mem_util=gpu_mem_util)
        _write_limits(lim_path_, lim)
        console.print("[green]  Tuned.[/green]")
    limits = lim.get("limits", {})

    # 1. Context window
    working_ctxs = sorted(
        (int(c) for c, d in limits.items() if any(v is not None for v in d.values())),
    )
    if not working_ctxs:
        console.print("[red]No working configs found in probe data.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Configure {m.model_name}[/bold] (vLLM)\n")
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

    # Tool calling & reasoning
    tool_info = backend.detect_tools(m.path)
    resolved_parser: str | None = None
    resolved_reasoning: str | None = None

    if tool_parser:
        resolved_parser = tool_parser
        tools = True
    else:
        resolved_parser = lim.get("tool_call_parser") or tool_info.get("tool_call_parser")

    resolved_reasoning = lim.get("reasoning_parser") or tool_info.get("reasoning_parser")

    if resolved_parser or resolved_reasoning:
        console.print("\n  [bold]5. Capabilities[/bold]")
        if resolved_parser:
            if not tools:
                enable_tools = typer.confirm(
                    f"     Enable tool calling? (parser: {resolved_parser})", default=False,
                )
                tools = enable_tools
            else:
                console.print(f"     Tool calling: [green]{resolved_parser}[/green]")
        if resolved_reasoning:
            if not tools:
                console.print(f"     Reasoning: [dim]{resolved_reasoning} (enable tool calling to activate)[/dim]")
            else:
                console.print(f"     Reasoning:    [green]{resolved_reasoning}[/green]")
    else:
        from vserve.tools import supports_tools as _supports_tools
        if _supports_tools(m.path):
            console.print("\n  [bold]5. Capabilities[/bold]")
            console.print("     [yellow]Tool markers found but parser unknown[/yellow]")
            console.print("     Use --tool-parser <name> to enable")

    # Build config via backend
    choices = {
        "context": chosen_ctx,
        "kv_dtype": chosen_kv,
        "slots": chosen_seqs,
        "batched_tokens": chosen_bt,
        "gpu_mem_util": gpu_mem_util,
        "port": 8888,
        "tools": tools and bool(resolved_parser),
        "tool_parser": resolved_parser if tools else None,
        "reasoning_parser": resolved_reasoning if tools else None,
    }
    cfg = backend.build_config(m, choices)

    # Summary
    console.print("\n  [bold]Summary[/bold]")
    console.print(f"    Context:        {chosen_ctx // 1024}k")
    console.print(f"    KV dtype:       {chosen_kv}")
    console.print(f"    Slots:          {chosen_seqs}")
    console.print(f"    Batched tokens: {chosen_bt or 'auto'}")
    console.print("    Prefix:         always on")
    if tools and resolved_parser:
        cap_parts = [f"tools=[green]{resolved_parser}[/green]"]
        if resolved_reasoning:
            cap_parts.append(f"reasoning=[green]{resolved_reasoning}[/green]")
        console.print(f"    Tool calling:   {' '.join(cap_parts)}")

    confirm = typer.prompt("\n  Start? [Y/n]", default="Y")
    if confirm.strip().lower() == "n":
        raise typer.Exit(0)

    cfg_path = profile_path(m.provider, m.model_name, "custom")
    write_profile_yaml(cfg_path, cfg, comment="vserve start — custom config")
    console.clear()
    return cfg_path


@app.command()
def start(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    tools: bool = typer.Option(False, "--tools", help="Enable tool/function calling"),
    tool_parser: str | None = typer.Option(None, "--tool-parser", help="Override tool-call parser (e.g. hermes, qwen3_coder)"),
    backend_name: str | None = typer.Option(None, "--backend", help="Force backend (vllm, llamacpp)"),
):
    """Start serving a model — interactive config picker."""
    from vserve.backends import get_backend, get_backend_by_name

    # Check session lock early — before interactive config
    _session_or_exit()

    if model is None:
        all_models = _all_models()
        if not all_models:
            console.print("[red]No models found.[/red] Run: vserve download")
            raise typer.Exit(1)

        while True:
            console.print("\n[bold]Select a model:[/bold]")
            for i, m in enumerate(all_models, 1):
                has_limits = read_limits(limits_path(m.provider, m.model_name)) is not None
                status = "[green]tuned[/green]" if has_limits else "[dim]run vserve tune first[/dim]"
                console.print(f"  {i}) {m.full_name}  ({m.model_size_gb} GB)  {status}")

            choice = _pick_number("\nModel number", len(all_models))
            m = all_models[choice - 1]
            if read_limits(limits_path(m.provider, m.model_name)) is not None:
                break
            console.print(f"[yellow]Run vserve tune {m.model_name} first.[/yellow]")
    else:
        m = _resolve_model(model)

    if backend_name:
        backend = get_backend_by_name(backend_name)
    else:
        backend = get_backend(m)

    if tool_parser and not tools:
        tools = True
    cfg_path = _custom_config(m, backend, tools=tools, tool_parser=tool_parser)
    _launch_backend(backend, cfg_path, m.model_name)


@app.command()
def stop():
    """Stop the inference server."""
    from vserve.backends import _BACKENDS

    _session_or_exit()

    # Find which backend is running
    running_backend = None
    for b in _BACKENDS:
        try:
            if b.is_running():
                running_backend = b
                break
        except Exception:
            continue

    if running_backend is None:
        clear_session()
        console.print("[dim]No server is running.[/dim]")
        return

    lock = _lock_or_exit("gpu", f"stopping {running_backend.display_name}")
    try:
        _session_or_exit()  # re-check under flock (TOCTOU)

        try:
            running_backend.stop()
        except RuntimeError as e:
            clear_session()
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        clear_session()
        console.print(f"[green]{running_backend.display_name} stopped.[/green]")
    finally:
        lock.release()


@app.command()
def fan(
    mode: str = typer.Argument(None, help="auto | <30-100> | off"),
):
    """GPU fan control — auto curve, fixed speed, or off."""
    from pathlib import Path

    import vserve.fan as _fan
    from vserve.fan import read_state
    from vserve.gpu import get_fan_speed
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
            if "vserve.fan" not in cmdline:
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
        from vserve.lock import wait_for_release
        if not wait_for_release("fan", timeout=5.0):
            console.print("[red]Old fan daemon did not release lock in time.[/red]")
            raise typer.Exit(1)
        import subprocess
        import sys
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "vserve.fan",
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
            from vserve.config import cfg as _cfg
            console.print(f"[red]Fan daemon failed to start.[/red] Check {_cfg().logs_dir / 'vserve-fan.log'}")
            raise typer.Exit(1)
        console.print(f"[green]Fan daemon started[/green] (pid {proc.pid})")

    def _start_fixed_daemon(speed: int) -> None:
        _stop_daemon()
        from vserve.lock import wait_for_release
        if not wait_for_release("fan", timeout=5.0):
            console.print("[red]Old fan daemon did not release lock in time.[/red]")
            raise typer.Exit(1)
        import subprocess
        import sys
        proc = subprocess.Popen(
            [sys.executable, "-m", "vserve.fan", "--fixed", str(speed)],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        if proc.poll() is not None:
            from vserve.config import cfg as _cfg
            console.print(f"[red]Fan daemon failed to start.[/red] Check {_cfg().logs_dir / 'vserve-fan.log'}")
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
        from vserve.gpu import restore_fan_auto
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
            from vserve.fan import _get_gpu_temp
            temp = _get_gpu_temp()
        except Exception:
            temp = None

        temp_str = f"{temp}°C" if temp is not None else "?"
        console.print(f"\n[bold]GPU Fan[/bold]  {speed}% @ {temp_str}")

        if _is_orphaned_fan():
            console.print("  [red]Warning: fan daemon crashed — fan may be stuck in manual mode[/red]")
            console.print("  [red]Run: sudo vserve fan off[/red]")
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
    """Show current serving status."""
    from vserve.backends import _BACKENDS
    from vserve.config import active_yaml_path, read_profile_yaml

    running_backend = None
    for b in _BACKENDS:
        try:
            if b.is_running():
                running_backend = b
                break
        except Exception:
            continue

    if running_backend is None:
        console.print("[dim]No server is running.[/dim] Start with: vserve start")
        return

    # Find active config — different path per backend
    import json as _json
    cfg = {}
    config_source = None
    if running_backend.name == "llamacpp":
        active_json = running_backend._active_config_path().with_suffix(".json")
        if active_json.exists():
            try:
                cfg = _json.loads(active_json.read_text())
                config_source = active_json
            except Exception:
                pass
    if not cfg:
        active = active_yaml_path()
        if active.exists():
            cfg = read_profile_yaml(active) or {}
            config_source = active.resolve() if active.is_symlink() else active
    if not cfg:
        console.print(f"[bold green]{running_backend.display_name} is running[/bold green] (no active config found)")
        return

    model_path = cfg.get("model", "?")
    model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path

    ctx = cfg.get("max-model-len") or cfg.get("ctx-size") or cfg.get("ctx_size", "?")
    ctx_display = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
    seqs = cfg.get("max-num-seqs") or cfg.get("parallel", "?")
    kv = cfg.get("kv-cache-dtype", "auto")
    prefix = cfg.get("enable-prefix-caching", False)
    bt = cfg.get("max-num-batched-tokens")
    port = cfg.get("port", 8888)
    gpu_util = cfg.get("gpu-memory-utilization")
    gpu_str = f"{gpu_util:.1%}" if isinstance(gpu_util, float) else str(gpu_util)

    console.print(f"\n[bold green]{running_backend.display_name} is running[/bold green]")
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
    if config_source:
        console.print(f"\n  [dim]{config_source}[/dim]\n")


@app.command()
def init():
    """Set up vserve — scan the system, write config, optionally install login banner."""
    import shutil
    import subprocess
    from pathlib import Path as _Path

    from vserve.config import (
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

    console.print("\n[bold]vserve setup[/bold]\n")
    console.print("[dim]Scanning your system...[/dim]\n")

    # ── GPU ──
    try:
        from vserve.gpu import get_gpu_info
        gpu = get_gpu_info()
        _ok(f"GPU         {gpu.name} ({gpu.vram_total_gb:.0f} GB)")
        _ok(f"Driver      {gpu.driver}")
        _ok(f"CUDA        {gpu.cuda}")
    except Exception:
        _fail("GPU         nvidia-smi not found or no GPU detected")
        _fail("            vserve requires an NVIDIA GPU with drivers installed")
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

    # ── llama.cpp ──
    llamacpp_root = None
    llamacpp_bin = shutil.which("llama-server")
    llamacpp_candidate = _Path("/opt/llama-cpp")
    if (llamacpp_candidate / "bin" / "llama-server").exists():
        llamacpp_root = llamacpp_candidate
        _ok(f"llama.cpp   found at {llamacpp_candidate}")
    elif llamacpp_bin:
        llamacpp_root = _Path(llamacpp_bin).resolve().parent.parent
        _ok(f"llama.cpp   {llamacpp_bin}")
    else:
        _warn("llama.cpp   not found (optional — for GGUF models)")

    # ── Models ──
    models_dir = root / "models"
    if models_dir.is_dir():
        models = list(models_dir.glob("*/*/config.json"))
        _ok(f"Models      {len(models)} found at {models_dir}")
    else:
        _warn(f"Models      {models_dir} does not exist")
    if llamacpp_root:
        lc_models = llamacpp_root / "models"
        if lc_models.is_dir():
            gguf_count = sum(1 for _ in lc_models.rglob("*.gguf"))
            _ok(f"GGUF models {gguf_count} found at {lc_models}")
        else:
            _warn(f"GGUF models {lc_models} does not exist")

    # ── systemd ──
    svc_name, svc_user = _discover_service()
    svc_path = _Path(f"/etc/systemd/system/{svc_name}.service")
    if svc_path.exists():
        _ok(f"systemd     {svc_name}.service (user: {svc_user})")
    else:
        _fail("systemd     no vLLM systemd service found (required for vserve start/stop)")
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
        _warn("Coolbits    not configured (vserve fan requires it)")

    # ── gum (interactive UI) ──
    if shutil.which("gum"):
        _ok("gum         installed (interactive UI enabled)")
    else:
        _warn("gum         not installed (vserve download will use basic mode)")

    # ── Write config ──
    console.print()
    if CONFIG_FILE.exists():
        console.print(f"  [dim]Config exists at {CONFIG_FILE}[/dim]")
        if not typer.confirm("  Overwrite config?", default=True):
            console.print()
        else:
            config = _build_config(root, cuda, svc_name, svc_user, port, llamacpp_root=llamacpp_root)
            save_config(config)
            reset_config()
            _ok(f"Config written to {CONFIG_FILE}")
    else:
        config = _build_config(root, cuda, svc_name, svc_user, port, llamacpp_root=llamacpp_root)
        save_config(config)
        reset_config()
        _ok(f"Config written to {CONFIG_FILE}")

    # ── Welcome banner ──
    banner_src = _Path(__file__).parent / "welcome.sh"
    banner_dest = CONFIG_FILE.parent / "welcome.sh"
    _marker = "# vserve login banner"

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
    console.print("  vserve download    Search & download a model")
    console.print("  vserve doctor      Full system check")
    console.print("  vserve             Dashboard")
    console.print()


@app.command()
def doctor():
    """Check system readiness."""
    import os
    import subprocess
    import socket
    from pathlib import Path

    from vserve.config import VLLM_BIN, VLLM_ROOT, active_yaml_path, read_profile_yaml, LOGS_DIR
    from vserve.backends import any_backend_running

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

    console.print("\n[bold]vserve doctor[/bold]\n")

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
        from vserve.gpu import get_gpu_info
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

    # -- Backends --
    console.print("\n  [bold]Backends[/bold]")
    from vserve.backends import _BACKENDS
    for b in _BACKENDS:
        for desc, check_fn in b.doctor_checks():
            try:
                if check_fn():
                    _ok(desc)
                else:
                    _warn(desc)
            except Exception:
                _warn(f"{desc} (check error)")

    # Service user
    from vserve.config import cfg as _cfg
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
              "Create one: https://docs.vllm.ai  or see vserve docs/troubleshooting.md")

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
            _warn(f"{len(stale)} stale sockets in {tmp_dir}", f"sudo find {tmp_dir} -type s -delete")
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
        _warn("No active.yaml — run vserve start to configure")

    # Port
    _port = _c.port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        port_in_use = s.connect_ex(("127.0.0.1", _port)) == 0
    if any_backend_running():
        _ok(f"Serving on port {_port}")
    elif port_in_use:
        _fail(f"Port {_port} in use but no backend running — something else is bound",
              f"Check: sudo lsof -i :{_port}  then stop the other service or change port in ~/.config/vserve/config.yaml")
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
            _warn(f"'{me}' not in tty group — vserve can't DM other users",
                  f"sudo usermod -aG tty {me}  (then re-login)")
    except KeyError:
        _warn("tty group not found")

    # Models
    all_models = _all_models()
    probed = sum(1 for m in all_models if read_limits(limits_path(m.provider, m.model_name)))
    _ok(f"{len(all_models)} models downloaded, {probed} probed")

    console.print(f"\n  [bold]{'All clear' if fail_count == 0 else f'{fail_count} issue(s) found'}[/bold]  "
                  f"({ok_count} ok, {fail_count} fail)\n")


# ── Cache management ──

@cache_app.command("clean")
def cache_clean(
    all_caches: bool = typer.Option(False, "--all", help="Also clean flashinfer, torch, vllm caches"),
):
    """Clean stale sockets and optionally JIT caches."""
    import subprocess
    from vserve.config import cfg as _cfg
    _c = _cfg()
    root = _c.vllm_root
    total_freed = 0

    # Always: stale sockets in tmp
    tmp_dir = root / "tmp"
    if tmp_dir.exists():
        sockets = [s for s in tmp_dir.iterdir() if s.is_socket()]
        if sockets:
            result = subprocess.run(
                ["sudo", "find", str(tmp_dir), "-type", "s", "-delete"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                console.print(f"  [green]Cleaned {len(sockets)} stale sockets from {tmp_dir}[/green]")
            else:
                console.print(f"  [red]Failed to clean sockets: {result.stderr.strip()}[/red]")
        else:
            console.print(f"  [dim]No stale sockets in {tmp_dir}[/dim]")

    if all_caches:
        cache_dirs = [
            (root / ".cache" / "flashinfer", "FlashInfer JIT"),
            (root / ".cache" / "torch_extensions", "torch compile"),
            (root / ".cache" / "vllm", "vLLM"),
        ]
        for cache_dir, label in cache_dirs:
            if cache_dir.exists():
                # Measure size before deleting
                result = subprocess.run(
                    ["sudo", "du", "-sm", str(cache_dir)],
                    capture_output=True, text=True,
                )
                size_mb = 0
                if result.returncode == 0:
                    try:
                        size_mb = int(result.stdout.split()[0])
                    except (ValueError, IndexError):
                        pass
                result = subprocess.run(
                    ["sudo", "rm", "-rf", str(cache_dir)],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    total_freed += size_mb
                    console.print(f"  [green]Cleaned {label} cache ({size_mb} MB)[/green]")
                else:
                    console.print(f"  [red]Failed to clean {label}: {result.stderr.strip()}[/red]")
            else:
                console.print(f"  [dim]{label} cache not found[/dim]")

        if total_freed > 0:
            console.print(f"\n  [bold]Freed {total_freed} MB total[/bold]")
        console.print("  [yellow]Next vserve start will recompile kernels (~5-10 min)[/yellow]")
