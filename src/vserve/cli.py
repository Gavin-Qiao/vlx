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
from vserve.lock import (
    VserveLock,
    LockHeld,
    SessionHeld,
    SessionUnknown,
    notify_user,
    check_session,
    write_session,
    clear_session,
)

from vserve import __version__

app = typer.Typer(help="LLM inference manager")
cache_app = typer.Typer(help="Cache management")
app.add_typer(cache_app, name="cache")
console = Console()
_TITLE = f"[bold cyan]vserve[/bold cyan] [dim]{__version__}[/dim]"



def _session_or_exit(
    *,
    fail_on_probe_uncertainty: bool = True,
    allow_unknown_owner: bool = False,
) -> None:
    """Block if another user owns the active GPU session, DM the holder."""
    import os
    try:
        check_session(fail_on_probe_uncertainty=fail_on_probe_uncertainty)
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
    except SessionUnknown as exc:
        if allow_unknown_owner:
            console.print(f"[yellow]{exc.message()}[/yellow]")
            return
        console.print(Panel(
            f"[bold]{exc.message()}[/bold]"
            "\n\nUse [cyan]vserve status[/cyan] or inspect the backend service before starting another model.",
            title=f"[red]vserve {__version__}: session owner unknown[/red]",
            border_style="red",
        ))
        raise typer.Exit(1) from None


def _safe_path_exists(path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_resolve_path(path):
    try:
        path.stat()
    except FileNotFoundError:
        try:
            return path.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
    except (OSError, RuntimeError):
        return None
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


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
        console.print("[red]No models found.[/red] Run: vserve add")
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
    from vserve.config import active_yaml_path, try_read_profile_yaml

    models = _all_models()
    probed = sum(
        1 for m in models if read_limits(limits_path(m.provider, m.model_name))
    )

    serving_line = "[dim]not running[/dim]"
    rb = _running_backend()
    if rb is not None:
        active = active_yaml_path()
        if active.is_symlink():
            cfg = try_read_profile_yaml(active)
            model_path = cfg.get("model", "?") if cfg else "?"
            model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path
            port = cfg.get("port", 8888) if cfg else 8888
            if cfg:
                serving_line = f"[green]{model_name}[/green] at :{port} ({rb.display_name})"
            else:
                serving_line = f"[yellow]config unreadable[/yellow] ({rb.display_name})"
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
        ("list", "List models with limits & capabilities"),
        ("add [model]", "Search & download from HuggingFace"),
        ("rm [model]", "Remove a downloaded model"),
        ("tune [model]", "Calculate context & concurrency limits"),
        ("run [model]", "Start serving (interactive config)"),
        ("stop", "Stop the inference server"),
        ("status", "Show what's currently serving"),
        ("fan [auto|off|30-100]", "GPU fan control with temp curve"),
        ("doctor", "Check system readiness"),
        ("init", "Auto-discover backends and write config"),
        ("version", "Show current version"),
        ("update", "Update to the latest version"),
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
def update(
    nightly: bool = typer.Option(False, "--nightly", help="Install latest pre-release version"),
):
    """Update vserve to the latest version."""
    import shutil
    import subprocess

    from vserve.version import check_pypi, write_cache, _compare_versions
    command_timeout = 300

    def _run_checked(cmd: list[str], label: str) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=command_timeout)
        except subprocess.TimeoutExpired:
            console.print(f"[red]{label} timed out after {command_timeout}s.[/red]")
            raise typer.Exit(1)
        if result.returncode != 0:
            console.print(f"[red]{label} failed.[/red]")
            details = result.stderr.strip() or result.stdout.strip()
            if details:
                console.print(f"  {details}")
            raise typer.Exit(1)
        return result

    console.print(f"[dim]Current version: {__version__}[/dim]")

    if not nightly:
        latest = check_pypi()
        if not latest:
            console.print("[red]Could not determine the latest published version.[/red]")
            console.print("  Retry later, or use [cyan]vserve update --nightly[/cyan] if you intentionally want a pre-release.")
            raise typer.Exit(1)
        write_cache(latest)
        if _compare_versions(latest, __version__) <= 0:
            console.print("[green]Already up to date.[/green]")
            return
        console.print(f"[yellow]New version available: {latest}[/yellow]\n")

    uv = shutil.which("uv")
    if uv:
        try:
            result = subprocess.run([uv, "tool", "list"], capture_output=True, text=True, timeout=command_timeout)
        except subprocess.TimeoutExpired:
            console.print(f"[yellow]uv inspection timed out after {command_timeout}s; trying pip fallback if available.[/yellow]")
            result = None
        if result is not None and result.returncode != 0:
            console.print("[yellow]uv inspection failed; trying pip fallback if available.[/yellow]")
            details = result.stderr.strip() or result.stdout.strip()
            if details:
                console.print(f"  {details}")
            result = None
        if result is not None and "vserve" in result.stdout:
            if nightly:
                console.print("[dim]Upgrading to latest pre-release via uv...[/dim]")
                _run_checked([uv, "tool", "upgrade", "vserve", "--prerelease", "allow"], "uv upgrade")
            else:
                console.print("[dim]Upgrading via uv...[/dim]")
                _run_checked([uv, "tool", "upgrade", "vserve"], "uv upgrade")
            _refresh_banner()
            return

    pip = shutil.which("pip") or shutil.which("pip3")
    if pip:
        import sys
        pip_cmd = [sys.executable, "-m", "pip"]
        if nightly:
            console.print("[dim]Upgrading to latest pre-release via pip...[/dim]")
            _run_checked([*pip_cmd, "install", "--upgrade", "--pre", "vserve"], "pip install")
        else:
            console.print("[dim]Upgrading via pip...[/dim]")
            _run_checked([*pip_cmd, "install", "--upgrade", "vserve"], "pip install")
        _refresh_banner()
        return

    console.print("[red]Could not find uv or pip to perform upgrade.[/red]")
    console.print("Run manually: [cyan]uv tool upgrade vserve[/cyan] or [cyan]pip install -U vserve[/cyan]")
    raise typer.Exit(1)


def _refresh_banner() -> None:
    """Silently update the login banner if installed."""
    from pathlib import Path as _P
    banner_src = _P(__file__).parent / "welcome.sh"
    banner_dest = CONFIG_FILE.parent / "welcome.sh"
    if banner_dest.exists() and banner_src.exists():
        import shutil
        shutil.copy2(banner_src, banner_dest)


@app.command(name="list")
def list_models(model: str = typer.Argument(None, help="Model name for detail view")):
    """List downloaded models."""
    from vserve.backends import _BACKENDS

    all_models = _all_models()
    if not all_models:
        console.print("[dim]No models found.[/dim] Run: vserve add")
        return

    if model:
        m = _resolve_model(model)
        _show_model_detail(m)
        return

    _backend_colors = {"vLLM": "cyan", "llama.cpp": "magenta"}

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="bold")
    table.add_column("Backend")
    table.add_column("Size", justify="right")
    table.add_column("Limits")
    table.add_column("Max Context", justify="right")
    table.add_column("Tools", style="green")
    table.add_column("Reasoning", style="green")

    for m in all_models:
        lim = read_limits(limits_path(m.provider, m.model_name))

        # Detect backend
        backend_name = "\u2014"
        backend_obj = None
        for b in _BACKENDS:
            if b.can_serve(m):
                backend_name = b.display_name
                backend_obj = b
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

        # Use cached limits for tool/reasoning info (fast) — only live-detect for untuned models
        tp = "\u2014"
        rp = "\u2014"
        if lim:
            # vLLM: parser names in limits
            tp = lim.get("tool_call_parser") or ("jinja" if lim.get("supports_tools") else "\u2014")
            rp = lim.get("reasoning_parser") or ("jinja" if lim.get("supports_reasoning") else "\u2014")
        elif backend_obj:
            # Untuned — live detect (slow for GGUF, but only for untuned models)
            tool_info = backend_obj.detect_tools(m.path)
            tp = tool_info.get("tool_call_parser") or ("jinja" if tool_info.get("supports_tools") else "\u2014")
            rp = tool_info.get("reasoning_parser") or ("jinja" if tool_info.get("supports_reasoning") else "\u2014")

        bc = _backend_colors.get(backend_name, "white")
        table.add_row(
            m.full_name, f"[{bc}]{backend_name}[/{bc}]", f"{m.model_size_gb} GB",
            "\u2713" if lim else "\u2717", max_ctx, tp, rp,
        )

    console.print(table)


@app.command(name="ls", hidden=True)
def ls(model: str = typer.Argument(None, help="Model name for detail view")):
    """List downloaded models (alias for list)."""
    list_models(model)


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


def _is_interactive() -> bool:
    """True when stdin is a real terminal (not CI, not CliRunner)."""
    import sys
    return sys.stdin.isatty()


def _has_gum() -> bool:
    import shutil
    return shutil.which("gum") is not None


def _restore_terminal() -> None:
    """Reset terminal to sane state after menu tools may have altered it."""
    import sys
    try:
        import termios
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


# Save terminal state at import time so we can restore after menus
try:
    import sys as _sys
    import termios as _termios
    _SAVED_TERM_ATTRS = _termios.tcgetattr(_sys.stdin.fileno())
except Exception:
    _SAVED_TERM_ATTRS = None  # type: ignore[assignment]

# Enable readline for input() — arrow keys, history, backspace
try:
    import readline as _readline  # noqa: F401
except ImportError:
    pass


def _pick(items: list[str], title: str = "") -> int | None:
    """Single-select menu. Returns index or None on cancel.

    Uses gum → simple-term-menu → numbered prompt (best available).
    """
    if not _is_interactive():
        # CI / CliRunner fallback
        for i, item in enumerate(items, 1):
            console.print(f"  {i}) {item}")
        while True:
            choice = typer.prompt(title or "Choice")
            try:
                n = int(choice)
                if 1 <= n <= len(items):
                    return n - 1
            except ValueError:
                pass
            console.print(f"[red]Enter a number 1-{len(items)}.[/red]")

    if _has_gum():
        import subprocess
        cmd = ["gum", "choose", "--cursor.foreground", "6", "--item.faint"]
        if title:
            cmd.extend(["--header", title])
        cmd.extend(items)
        r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        _restore_terminal()
        if r.returncode != 0 or not r.stdout.strip():
            return None
        selected = r.stdout.strip()
        for i, item in enumerate(items):
            if item == selected:
                return i
        return None

    try:
        from simple_term_menu import TerminalMenu
        menu = TerminalMenu(
            items, title=title,
            menu_cursor="❯ ",
            menu_cursor_style=("fg_cyan", "bold"),
            menu_highlight_style=("standout",),
            cycle_cursor=True,
            status_bar="  ↑↓ navigate · enter select · q cancel",
            status_bar_style=("fg_gray",),
        )
        idx = menu.show()
        _restore_terminal()
        return idx  # type: ignore[return-value]
    except Exception:
        _restore_terminal()

    # Final fallback: numbered prompt
    for i, item in enumerate(items, 1):
        console.print(f"  {i}) {item}")
    while True:
        choice = typer.prompt(title or "Choice")
        try:
            n = int(choice)
            if 1 <= n <= len(items):
                return n - 1
        except ValueError:
            pass
        console.print(f"[red]Enter a number 1-{len(items)}.[/red]")


def _parse_multi_select(answer: str, item_count: int) -> list[int] | None:
    """Parse comma/space-separated menu indices.

    Returns None when any token is invalid so callers can re-prompt instead of
    silently dropping part of the user's selection.
    """
    text = answer.strip()
    if not text:
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for part in text.replace(",", " ").split():
        try:
            n = int(part)
        except ValueError:
            return None
        if not 1 <= n <= item_count:
            return None
        idx = n - 1
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _pick_many(items: list[str], title: str = "") -> list[int]:
    """Multi-select menu. Returns list of indices.

    Uses gum → simple-term-menu → numbered prompt (best available).
    """
    if not _is_interactive():
        for i, item in enumerate(items, 1):
            console.print(f"  {i}) {item}")
        while True:
            answer = typer.prompt(title or "Select (e.g. 1 or 1,2; blank cancels)", default="")
            parsed = _parse_multi_select(answer, len(items))
            if parsed is not None:
                return parsed
            console.print(
                f"[red]Enter one or more numbers 1-{len(items)}, separated by commas or spaces.[/red]",
            )

    if _has_gum():
        import subprocess
        cmd = [
            "gum", "choose", "--no-limit",
            "--cursor.foreground", "6",
            "--selected.foreground", "2", "--selected.bold",
        ]
        if title:
            cmd.extend(["--header", title])
        cmd.extend(items)
        r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
        _restore_terminal()
        if r.returncode != 0 or not r.stdout.strip():
            return []  # cancelled or nothing selected → back
        selected_lines = {ln.strip() for ln in r.stdout.strip().split("\n")}
        return [i for i, item in enumerate(items) if item.strip() in selected_lines]

    try:
        from simple_term_menu import TerminalMenu
        menu = TerminalMenu(
            items, title=title,
            multi_select=True,
            show_multi_select_hint=True,
            menu_cursor="❯ ",
            menu_cursor_style=("fg_cyan", "bold"),
            multi_select_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("standout",),
            cycle_cursor=True,
            status_bar="  ↑↓ navigate · x/space toggle · enter confirm · q cancel",
            status_bar_style=("fg_gray",),
        )
        result = menu.show()
        _restore_terminal()
        if result is None:
            return []
        return list(result) if isinstance(result, tuple) else [result]
    except Exception:
        _restore_terminal()
        pass

    # Final fallback: comma-separated prompt
    for i, item in enumerate(items, 1):
        console.print(f"  {i}) {item}")
    while True:
        answer = typer.prompt(title or "Select (e.g. 1 or 1,2; blank cancels)", default="")
        parsed = _parse_multi_select(answer, len(items))
        if parsed is not None:
            return parsed
        console.print(
            f"[red]Enter one or more numbers 1-{len(items)}, separated by commas or spaces.[/red]",
        )


@app.command()
def add(model_id: str = typer.Argument(None, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-27B-FP8)")):
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

    # Search loop
    query = model_id or ""
    results: list = []
    while True:
        if not query:
            try:
                console.print()
                query = input("  Search HuggingFace (Ctrl-C to quit): ")
            except (KeyboardInterrupt, EOFError):
                console.print()
                return
            if not query:
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

        lines = []
        for m in results:
            dl = f"{m.downloads:,}" if m.downloads else "?"
            lines.append(f"{m.id}  ({dl} downloads)")

        idx = _pick(lines, title=f"Results for '{query}':")
        if idx is None:
            console.clear()
            query = ""
            results = []
            continue

        downloaded = _download_model(results[idx].id, models_dir, snapshot_download, api)
        if not downloaded:
            console.clear()
            continue  # backed out of variant picker — re-show results
        # Offer to download another
        if not typer.confirm("\n  Download another model?", default=False):
            return
        console.clear()
        query = ""
        results = []


def _pick_variants(variants: list) -> list:
    """Interactive variant selection. Returns list of selected Variant objects."""
    from vserve.variants import format_variant_line

    # Single variant — auto-select, no picker needed
    if len(variants) == 1:
        console.print(f"  {format_variant_line(variants[0], index=1)}")
        return variants

    lines = [format_variant_line(v, index=i) for i, v in enumerate(variants, 1)]
    indices = _pick_many(lines, title="Select variant(s) — x/space to toggle, enter to confirm:")
    return [variants[i] for i in indices]


def _download_model(model_id: str, models_dir: "pathlib.Path", snapshot_download: object, api: object) -> bool:
    """Download a single model by its HuggingFace ID. Returns True if download happened."""
    from vserve.variants import fetch_repo_variants, _format_bytes

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

    # Selection (variants shown inside _pick_variants via _pick_many)
    selected_variants = _pick_variants(variants)
    if not selected_variants:
        console.clear()
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
        if info.max_position_embeddings:
            console.print(f"  Context: {info.max_position_embeddings:,} tokens")

        # Auto-tune after download
        try:
            from vserve.backends import get_backend
            from vserve.gpu import get_gpu_info, compute_gpu_memory_utilization
            from vserve.config import write_limits
            backend = get_backend(info)
            gpu = get_gpu_info()
            gpu_mem_util = compute_gpu_memory_utilization(gpu.vram_total_gb)
            console.print(f"\n[dim]Auto-tuning for {gpu.name}...[/dim]")
            limits_data = backend.tune(info, gpu, gpu_mem_util=gpu_mem_util)
            lim_path = limits_path(info.provider, info.model_name)
            write_limits(lim_path, limits_data)
            console.print(f"[green]Tuned.[/green] Ready: vserve run {model_name}")
        except Exception as e:
            console.print(f"  [yellow]Auto-tune skipped: {e}[/yellow]")
            console.print(f"  Run: vserve tune {model_name}")
    else:
        console.print(f"\n[green]Downloaded to {local_dir}[/green]")
    return True


def _model_rm_impl(model: str | None, force: bool = False):
    """Shared implementation for model rm / model remove."""
    import shutil

    if model is None:
        all_models = _all_models()
        if not all_models:
            console.print("[dim]No models to remove.[/dim]")
            return
        name_w = max(len(m.full_name) for m in all_models)
        size_w = max(len(f"{m.model_size_gb} GB") for m in all_models)
        items = [
            f"{m.full_name:<{name_w}}  {f'{m.model_size_gb} GB':>{size_w}}"
            for m in all_models
        ]
        idx = _pick(items, title="Remove which model?")
        if idx is None:
            return
        m = all_models[idx]
    else:
        m = _resolve_model(model)

    size_gb = m.model_size_gb
    console.print(f"\n  [bold]{m.full_name}[/bold]")
    console.print(f"  Path: {m.path}")
    console.print(f"  Size: {size_gb} GB\n")

    if not force:
        if not typer.confirm(f"  Delete {m.full_name}?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    # Delete model directory
    shutil.rmtree(m.path)
    console.print(f"[green]Deleted {m.full_name}[/green] ({size_gb} GB)")

    # Clean up cached limits
    lim = limits_path(m.provider, m.model_name)
    if lim.exists():
        lim.unlink()
        console.print("  [dim]Removed limits cache[/dim]")

    # Clean up profile configs
    prof = profile_path(m.provider, m.model_name, "custom")
    if prof.exists():
        prof.unlink()
        console.print("  [dim]Removed profile config[/dim]")

    # Remove empty parent dir (provider dir) if it's now empty
    parent = m.path.parent
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


@app.command()
def rm(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a downloaded model."""
    _model_rm_impl(model, force=force)


@app.command(hidden=True)
def remove(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Remove a downloaded model (alias for rm)."""
    _model_rm_impl(model, force=force)


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
            console.print("[red]No models found.[/red] Run: vserve add")
            raise typer.Exit(1)
        items = [m.full_name for m in all_m] + ["All models"]
        idx = _pick(items, title="Which model?")
        if idx is None:
            raise typer.Exit(0)
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
            # llama.cpp: tools/reasoning supported via --jinja, no parser name needed
            caps = ["tool calling"]
            if limits_data.get("supports_reasoning"):
                caps.append("reasoning")
            console.print(f"  Capabilities: [green]{' + '.join(caps)} (--jinja)[/green]")
        else:
            from vserve.tools import supports_tools as _supports_tools
            if _supports_tools(m.path):
                console.print("  Capabilities: [yellow]tool markers found but parser unknown[/yellow]")
                console.print("                use --tools --tool-parser <parser> with vserve run")
            else:
                console.print("  Capabilities: [dim]no tool calling detected[/dim]")
        console.print(f"  [green]Saved to {lim_path}[/green]")

        # Offer pre-caching for vLLM (flashinfer JIT compilation)
        if backend.name == "vllm" and not model and not all_models:
            # Only offer pre-cache in interactive mode (bare `vserve tune`)
            try:
                from vserve.config import cfg as _cfg2
                fi_cache = _cfg2().vllm_root / ".cache" / "flashinfer"
                needs_precache = not fi_cache.is_dir() or not any(fi_cache.rglob("*.so"))
            except Exception:
                needs_precache = False
            if needs_precache:
                console.print("  [yellow]First vLLM start will compile flashinfer kernels (~5-10 min).[/yellow]")
                if typer.confirm("  Pre-cache now? (starts and stops vLLM once)", default=False):
                    console.print("  [dim]Pre-caching — this will take a few minutes...[/dim]")
                    lock = _lock_or_exit("gpu", f"pre-caching {backend.display_name}")
                    started_here = False
                    try:
                        from vserve.config import write_profile_yaml
                        # Build a minimal config just for pre-caching
                        precache_cfg = backend.build_config(m, {
                            "context": 4096, "kv_dtype": "auto", "slots": 1,
                            "batched_tokens": None, "gpu_mem_util": gpu_util,
                            "port": 8888, "tools": False, "tool_parser": None,
                            "reasoning_parser": None,
                        })
                        cfg_path = profile_path(m.provider, m.model_name, "precache")
                        write_profile_yaml(cfg_path, precache_cfg, comment="vserve tune — pre-cache")
                        _session_or_exit()
                        try:
                            if backend.is_running():
                                console.print("  [yellow]vLLM is already running — skipping pre-cache warmup.[/yellow]")
                                continue
                        except Exception as e:
                            console.print(f"  [yellow]Could not determine backend state for pre-cache: {e}[/yellow]")
                            continue
                        backend.start(cfg_path)
                        started_here = True
                        write_session(f"{m.full_name} precache")
                        import time
                        from urllib.request import urlopen
                        health = backend.health_url(8888)
                        for i in range(150):
                            time.sleep(2)
                            try:
                                resp = urlopen(health, timeout=2)
                                if resp.status == 200:
                                    console.print("  [green]Pre-cache complete.[/green]")
                                    break
                            except Exception:
                                pass
                            if i > 5 and not backend.is_running():
                                console.print("  [yellow]Pre-cache failed — will compile on first real start.[/yellow]")
                                break
                            if i > 0 and i % 15 == 0:
                                console.print(f"  [dim]compiling... ({i * 2}s)[/dim]")
                    except Exception as e:
                        console.print(f"  [yellow]Pre-cache failed: {e}[/yellow]")
                    finally:
                        if started_here:
                            try:
                                backend.stop()
                            except Exception as e:
                                console.print(f"  [yellow]Could not stop pre-cache backend cleanly: {e}[/yellow]")
                            clear_session()
                        lock.release()
        console.print()


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
    import subprocess
    import time
    from vserve.backends import _BACKENDS, probe_running_backends
    from vserve.config import read_profile_yaml

    lock = _lock_or_exit("gpu", f"starting {backend.display_name} ({label})")

    # Read config (YAML for vLLM, JSON for llama.cpp)
    needs_precache = False
    health_timeout_s = 300
    health_poll_s = 3
    try:
        def _service_running_state() -> bool | None:
            try:
                return backend.is_running()
            except Exception:
                return None

        def _recent_service_log_tail() -> str | None:
            try:
                result = subprocess.run(
                    [
                        "journalctl",
                        "-u",
                        backend.service_name,
                        "--no-pager",
                        "-n",
                        "5",
                        "-o",
                        "cat",
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=1,
                )
            except Exception:
                return None
            if result.returncode != 0:
                return None
            tail = result.stdout.strip()
            return tail or None

        if str(cfg_path).endswith(".json"):
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception as exc:
                console.print(f"[red]Launch config unreadable:[/red] {exc}")
                console.print(f"  Config: {cfg_path}")
                raise typer.Exit(1)
            if not isinstance(cfg, dict):
                console.print("[red]Launch config unreadable:[/red] expected a JSON object")
                console.print(f"  Config: {cfg_path}")
                raise typer.Exit(1)
        else:
            try:
                cfg = read_profile_yaml(cfg_path) or {}
            except ValueError as exc:
                console.print(f"[red]Launch config unreadable:[/red] {exc}")
                console.print(f"  Config: {cfg_path}")
                raise typer.Exit(1)

        _session_or_exit(fail_on_probe_uncertainty=True)

        running_backends, probe_failed = probe_running_backends()
        if not running_backends and probe_failed:
            console.print("[red]Could not determine whether another backend is already running.[/red]")
            console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 20")
            raise typer.Exit(1)

        if running_backends:
            stop_candidates = list(running_backends)
            running_names = ", ".join(candidate.display_name for candidate in running_backends)
            same_backend_only = (
                len(running_backends) == 1
                and getattr(running_backends[0], "name", None) == backend.name
            )
            if probe_failed:
                stop_candidates = list(_BACKENDS)
                running_names = ", ".join(candidate.display_name for candidate in stop_candidates)
                console.print("[yellow]Backend state is uncertain; stopping all known backends before restart.[/yellow]")
                prompt = f"Stop {running_names} and start {backend.display_name}?"
            elif same_backend_only:
                console.print(f"[yellow]{backend.display_name} is already running.[/yellow]")
                prompt = "Stop and restart?"
            else:
                console.print(f"[yellow]Running backend(s): {running_names}.[/yellow]")
                prompt = f"Stop {running_names} and start {backend.display_name}?"
            if not typer.confirm(prompt, default=False):
                return
            for candidate in stop_candidates:
                try:
                    candidate.stop()
                except Exception as e:
                    console.print(f"[red]{e}[/red]")
                    console.print(f"  Check: sudo journalctl -u {candidate.service_name} --no-pager -n 20")
                    raise typer.Exit(1)
            for _ in range(15):
                time.sleep(2)
                remaining_backends, remaining_probe_failed = probe_running_backends()
                if not remaining_backends and not remaining_probe_failed:
                    break
            else:
                console.print("[red]Timed out waiting for running backends to stop before restart.[/red]")
                console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
                raise typer.Exit(1)

        ctx = cfg.get("max-model-len") or cfg.get("ctx_size") or cfg.get("ctx-size", "?")
        ctx_d = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        console.print(f"\n[bold]Starting[/bold] with [cyan]{label}[/cyan] ({backend.display_name})")
        console.print(f"  context: {ctx_d}")

        if backend.name == "vllm":
            from vserve.config import cfg as _cfg
            fi_cache = _cfg().vllm_root / ".cache" / "flashinfer"
            needs_precache = not fi_cache.is_dir() or not list(fi_cache.glob("**/*.so"))
            if needs_precache:
                health_timeout_s = 600
                console.print("  [yellow]First run — compiling kernels (~5-10 min).[/yellow]")

        try:
            backend.start(cfg_path)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 20")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Failed to start {backend.display_name}: {e}[/red]")
            console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 20")
            raise typer.Exit(1)

        write_session(label)

        from urllib.request import urlopen
        port = cfg.get("port", 8888)
        health = backend.health_url(port)

        console.print(f"  [dim]Waiting for {health} ...[/dim]")
        for i in range(max(1, (health_timeout_s + health_poll_s - 1) // health_poll_s)):
            time.sleep(health_poll_s)
            elapsed_s = (i + 1) * health_poll_s
            try:
                with urlopen(health, timeout=2) as resp:
                    if resp.status == 200:
                        console.print(f"\n[bold green]{backend.display_name} is running[/bold green] at http://localhost:{port}/v1")
                        console.print(f"  Config: {cfg_path}")
                        console.print(f"  Logs:   sudo journalctl -u {backend.service_name} -f\n")
                        return
            except Exception:
                pass

            log_tail = _recent_service_log_tail()
            if log_tail:
                console.print("  [dim]Latest service logs:[/dim]")
                for line in log_tail.splitlines():
                    console.print(f"    [dim]{line}[/dim]")

            if elapsed_s > 10:
                service_running = _service_running_state()
                if service_running is False:
                    clear_session()
                    console.print(f"[red]Service stopped unexpectedly.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
                    raise typer.Exit(1)

            if elapsed_s % 30 == 0:
                console.print(f"  [dim]still starting... ({elapsed_s}s)[/dim]")

        service_running = _service_running_state()

        if service_running is False:
            clear_session()
            console.print(f"[red]Timed out waiting for health endpoint.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
            raise typer.Exit(1)
        if service_running is True:
            console.print("[yellow]Health endpoint is still warming up, but the service remains active.[/yellow]")
            if needs_precache:
                console.print("  [yellow]First run kernel compilation may still be in progress.[/yellow]")
            if not _is_interactive():
                console.print("  [yellow]Non-interactive run requires the API health check to pass.[/yellow]")
                console.print(f"  Logs:   sudo journalctl -u {backend.service_name} -f")
                console.print(f"  Health: {health}")
                raise typer.Exit(1)
            console.print(f"  Returning while startup finishes in the background for {label}.")
            console.print(f"  Logs:   sudo journalctl -u {backend.service_name} -f")
            console.print(f"  Health: {health}")
            return
        console.print(f"[red]Timed out waiting for health endpoint, and service state could not be confirmed.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
        raise typer.Exit(1)
    finally:
        lock.release()



def _custom_config(m: ModelInfo, backend, *, tools: bool = False, tool_parser: str | None = None) -> "pathlib.Path":
    """Guide user through parameter selection, delegate config building to backend."""
    if backend.name == "llamacpp":
        return _custom_config_llamacpp(m, backend, tools=tools)
    return _custom_config_vllm(m, backend, tools=tools, tool_parser=tool_parser)


def _llamacpp_slots_from_limits_entry(entry: object) -> int | None:
    if isinstance(entry, dict):
        values = [v for v in entry.values() if v is not None]
        return max(values) if values else None
    if entry is None or isinstance(entry, bool) or not isinstance(entry, int):
        return None
    return entry


def _vllm_limits_entry(entry: object) -> dict[str, int | None]:
    if isinstance(entry, dict):
        cleaned: dict[str, int | None] = {}
        for key, value in entry.items():
            if not isinstance(key, str):
                continue
            if value is None or (isinstance(value, int) and not isinstance(value, bool)):
                cleaned[key] = value
        return cleaned
    if entry is None or isinstance(entry, bool) or not isinstance(entry, int):
        return {}
    return {"auto": entry}


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
    working_ctxs = sorted(
        int(str(c))
        for c, v in limits.items()
        if _llamacpp_slots_from_limits_entry(v) is not None
    )
    if not working_ctxs:
        console.print("[red]No working configs found.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Configure {m.model_name}[/bold] (llama.cpp)\n")
    ctx_items = []
    for ctx in working_ctxs:
        slots = _llamacpp_slots_from_limits_entry(limits.get(str(ctx)))
        slot_str = f"({slots} slots)" if slots else ""
        ctx_items.append(f"{ctx // 1024}k  {slot_str}")
    console.print("  [bold]1. Context window[/bold]")
    ctx_idx = _pick(ctx_items, title="  Context:")
    if ctx_idx is None:
        raise typer.Exit(0)
    chosen_ctx = working_ctxs[ctx_idx]

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
    max_parallel = _llamacpp_slots_from_limits_entry(limits.get(str(chosen_ctx), 1)) or 1
    console.print(f"\n  [bold]3. Parallel slots[/bold] (max: {max_parallel})")
    par_str = typer.prompt(f"  Slots [1-{max_parallel}]", default=str(max_parallel))
    try:
        chosen_parallel = min(max(1, int(par_str)), max_parallel)
    except ValueError:
        chosen_parallel = max_parallel

    # 4. Embedding or tool calling
    is_embedding = lim.get("is_embedding", False) or m.is_embedding
    chosen_pooling: str | None = None

    if is_embedding:
        pooling_options = ["mean", "cls", "last"]
        default_pooling = lim.get("pooling", "mean")
        default_idx = pooling_options.index(default_pooling) if default_pooling in pooling_options else 0
        pooling_labels = [
            "mean  — average all tokens (Nomic, E5, Jina, Qwen)",
            "cls   — [CLS] token only (BGE, BERT-style)",
            "last  — last token (decoder-based embeddings)",
        ]
        console.print("\n  [bold]4. Pooling strategy[/bold]")
        pool_idx = _pick(pooling_labels, title="  Pooling:")
        if pool_idx is None:
            pool_idx = default_idx
        chosen_pooling = pooling_options[pool_idx]
    else:
        # Use cached limits first, fall back to live detection
        supports_tools = lim.get("supports_tools", False)
        supports_reasoning = lim.get("supports_reasoning", False)
        if not supports_tools and not supports_reasoning:
            tool_info = backend.detect_tools(m.path)
            supports_tools = tool_info.get("supports_tools", False)
            supports_reasoning = tool_info.get("supports_reasoning", False)

        if supports_tools or supports_reasoning:
            caps = []
            if supports_tools:
                caps.append("tool calling")
            if supports_reasoning:
                caps.append("reasoning")
            console.print(f"\n  [bold]4. Capabilities[/bold] ({' + '.join(caps)} via --jinja)")
            if not tools:
                tools = typer.confirm("     Enable? (--jinja)", default=False)
            else:
                console.print("     [green]Enabled[/green] (--jinja)")

    # Build config via backend
    choices: dict = {
        "context": chosen_ctx,
        "n_gpu_layers": n_gpu_layers,
        "parallel": chosen_parallel,
        "port": 8888,
    }
    if is_embedding:
        choices["embedding"] = True
        choices["pooling"] = chosen_pooling
    else:
        choices["tools"] = tools
    cfg = backend.build_config(m, choices)

    # Summary
    console.print("\n  [bold]Summary[/bold]")
    console.print(f"    Context:     {chosen_ctx // 1024}k")
    console.print(f"    GPU layers:  {n_gpu_layers}/{num_layers}")
    console.print(f"    Parallel:    {chosen_parallel}")
    if is_embedding:
        console.print(f"    Mode:        [green]embedding (--pooling {chosen_pooling})[/green]")
    elif tools:
        console.print("    Tool calling: [green]enabled (--jinja)[/green]")

    if not typer.confirm("\n  Start?", default=True):
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
        (int(str(c)) for c, d in limits.items() if any(v is not None for v in _vllm_limits_entry(d).values())),
    )
    if not working_ctxs:
        console.print("[red]No working configs found in probe data.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Configure {m.model_name}[/bold] (vLLM)\n")

    # 1. Context window
    ctx_items = [f"{ctx // 1024}k" for ctx in working_ctxs]
    console.print("  [bold]1. Context window[/bold] (max tokens per request)")
    ctx_idx = _pick(ctx_items, title="  Context:")
    if ctx_idx is None:
        raise typer.Exit(0)
    chosen_ctx = working_ctxs[ctx_idx]

    # 2. KV cache dtype
    ctx_entry = _vllm_limits_entry(limits.get(str(chosen_ctx), {}))
    working_kvs = [k for k, v in ctx_entry.items() if v is not None]

    kv_labels = {
        "auto": "auto (bfloat16)",
        "fp8": "fp8 (saves memory, slightly less accurate)",
    }
    kv_items = [f"{kv}  — {kv_labels.get(kv, kv)}" for kv in working_kvs]
    console.print("\n  [bold]2. KV cache dtype[/bold]")
    kv_idx = _pick(kv_items, title="  KV dtype:")
    if kv_idx is None:
        raise typer.Exit(0)
    chosen_kv = working_kvs[kv_idx]

    # 3. Concurrent slots
    max_seqs = int(ctx_entry.get(chosen_kv, 1) or 1)
    console.print(f"\n  [bold]3. Concurrent slots[/bold] (max: {max_seqs})")
    console.print("     How many simultaneous requests?")
    seqs_str = typer.prompt(f"  Slots [1-{max_seqs}]", default=str(max_seqs))
    try:
        chosen_seqs = min(max(1, int(seqs_str)), max_seqs)
    except ValueError:
        chosen_seqs = max_seqs

    # 4. Batched tokens (for throughput tuning)
    bt_options = [
        "auto  — let vLLM decide (good for chat)",
        "2048  — balanced",
        "4096  — high throughput",
        "8192  — maximum throughput (batch processing)",
    ]
    bt_values: list[int | None] = [None, 2048, 4096, 8192]
    console.print("\n  [bold]4. Max batched tokens[/bold]")
    bt_idx = _pick(bt_options, title="  Batched tokens:")
    chosen_bt = bt_values[bt_idx] if bt_idx is not None else None

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

    if not typer.confirm("\n  Start?", default=True):
        raise typer.Exit(0)

    cfg_path = profile_path(m.provider, m.model_name, "custom")
    write_profile_yaml(cfg_path, cfg, comment="vserve run — custom config")
    console.clear()
    return cfg_path


@app.command()
def run(
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
            console.print("[red]No models found.[/red] Run: vserve add")
            raise typer.Exit(1)

        from vserve.backends import _BACKENDS

        name_w = max(len(m.full_name) for m in all_models)
        size_w = max(len(f"{m.model_size_gb} GB") for m in all_models)
        items = []
        for m in all_models:
            lim = read_limits(limits_path(m.provider, m.model_name))
            size = f"{m.model_size_gb} GB"

            # Capability tags
            tags: list[str] = []
            if lim:
                tags.append("tuned")
                tp = lim.get("tool_call_parser") or lim.get("supports_tools")
                rp = lim.get("reasoning_parser") or lim.get("supports_reasoning")
                if tp:
                    tags.append("tools")
                if rp:
                    tags.append("reasoning")
            else:
                # Try live detect for untuned models
                backend_obj = None
                for b in _BACKENDS:
                    if b.can_serve(m):
                        backend_obj = b
                        break
                if backend_obj:
                    tool_info = backend_obj.detect_tools(m.path)
                    if tool_info.get("tool_call_parser") or tool_info.get("supports_tools"):
                        tags.append("tools")
                    if tool_info.get("reasoning_parser") or tool_info.get("supports_reasoning"):
                        tags.append("reasoning")

            if m.is_embedding:
                tags.append("embedding")

            tag_str = "  ".join(tags) if tags else "not tuned"
            items.append(f"{m.full_name:<{name_w}}  {size:>{size_w}}  {tag_str}")

        idx = _pick(items, title="Select a model:")
        if idx is None:
            raise typer.Exit(0)
        m = all_models[idx]
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
    from vserve.backends import _BACKENDS, probe_running_backends

    _session_or_exit(fail_on_probe_uncertainty=False, allow_unknown_owner=True)

    running_backends, probe_failed = probe_running_backends()
    if not running_backends:
        if probe_failed:
            from vserve.backends import _BACKENDS

            lock = _lock_or_exit("gpu", "stopping backend (probe uncertain)")
            try:
                _session_or_exit(fail_on_probe_uncertainty=False, allow_unknown_owner=True)
                stop_errors: list[str] = []
                for candidate in _BACKENDS:
                    try:
                        candidate.stop()
                    except Exception as exc:
                        stop_errors.append(f"{candidate.display_name}: {exc}")
                confirmed_backends, confirmed_probe_failed = probe_running_backends()
                if confirmed_backends:
                    remaining_names = ", ".join(candidate.display_name for candidate in confirmed_backends)
                    console.print(f"[red]{remaining_names} still appear to be running after fallback stop attempts.[/red]")
                    for detail in stop_errors:
                        console.print(f"  {detail}")
                    raise typer.Exit(1)
                if confirmed_probe_failed:
                    console.print("[red]Could not verify backend state after fallback stop attempts.[/red]")
                    for detail in stop_errors:
                        console.print(f"  {detail}")
                    raise typer.Exit(1)
                clear_session()
                console.print("[yellow]Backend state was uncertain; issued stop requests to all known backends.[/yellow]")
                for detail in stop_errors:
                    console.print(f"  [dim]{detail}[/dim]")
                console.print("[green]Stop request completed.[/green]")
                return
            finally:
                lock.release()
        clear_session()
        console.print("[dim]No server is running.[/dim]")
        return

    stop_candidates = list(_BACKENDS) if probe_failed else list(running_backends)
    running_names = ", ".join(candidate.display_name for candidate in stop_candidates)
    lock = _lock_or_exit("gpu", f"stopping {running_names}")
    try:
        _session_or_exit(fail_on_probe_uncertainty=False, allow_unknown_owner=True)  # re-check under flock (TOCTOU)

        stop_errors: list[str] = []
        for candidate in stop_candidates:
            try:
                candidate.stop()
            except Exception as e:
                stop_errors.append(f"{candidate.display_name}: {e}")

        confirmed_backends, confirmed_probe_failed = probe_running_backends()
        if confirmed_backends:
            remaining_names = ", ".join(candidate.display_name for candidate in confirmed_backends)
            console.print(f"[red]{remaining_names} still appear to be running after stop.[/red]")
            for detail in stop_errors:
                console.print(f"  {detail}")
            raise typer.Exit(1)
        if confirmed_probe_failed:
            console.print("[red]Could not verify backend state after stop.[/red]")
            for detail in stop_errors:
                console.print(f"  {detail}")
            raise typer.Exit(1)
        clear_session()
        if stop_errors:
            for detail in stop_errors:
                console.print(f"  [dim]{detail}[/dim]")
        console.print(f"[green]Stopped: {running_names}.[/green]")
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

    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _daemon_pid() -> int | None:
        if not PID_PATH.exists():
            return None
        try:
            pid = int(PID_PATH.read_text().strip())
            os.kill(pid, 0)
            # Verify it's actually our daemon, not a recycled PID
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
            if "vserve.fan" not in cmdline:
                _safe_unlink(PID_PATH)
                _safe_unlink(STATE_PATH)
                return None
            return pid
        except (ValueError, OSError):
            _safe_unlink(PID_PATH)
            _safe_unlink(STATE_PATH)
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
        _safe_unlink(PID_PATH)
        _safe_unlink(STATE_PATH)
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
        _safe_unlink(_fan.PID_PATH)
        _safe_unlink(_fan.STATE_PATH)
        from vserve.gpu import restore_fan_auto
        try:
            restore_fan_auto()
            console.print("[green]Fan auto control restored.[/green]")
        except Exception:
            console.print("[yellow]Could not restore auto fan control via NVML.[/yellow]")

    def _show_status() -> None:
        try:
            speed: int | str = get_fan_speed()
        except Exception:
            speed = "?"
        pid = _daemon_pid()
        state = read_state() if pid else None
        try:
            from vserve.fan import _get_gpu_temp
            temp = _get_gpu_temp()
        except Exception:
            temp = None

        temp_str = f"{temp}°C" if temp is not None else "?"
        speed_str = f"{speed}%" if isinstance(speed, int) else "?"
        console.print(f"\n[bold]GPU Fan[/bold]  {speed_str} @ {temp_str}")

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
    descs = [desc for _, desc in options]
    idx = _pick(descs, title="Fan mode:")
    if idx is None:
        return
    action = options[idx][0]

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
    defaults = state if state and "fixed" not in state else {"quiet_start": 9, "quiet_end": 18, "quiet_max": 60}
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
    from vserve.config import active_yaml_path, try_read_profile_yaml

    running_backend = None
    for b in _BACKENDS:
        try:
            if b.is_running():
                running_backend = b
                break
        except Exception:
            continue

    if running_backend is None:
        console.print("[dim]No server is running.[/dim] Start with: vserve run")
        return

    # Find active config — different path per backend
    import json as _json
    cfg = {}
    config_source = None
    invalid_source = None
    if running_backend.name == "llamacpp":
        active_json = running_backend._active_config_path().with_suffix(".json")
        if active_json.exists():
            try:
                data = _json.loads(active_json.read_text())
                if isinstance(data, dict):
                    cfg = data
                    config_source = active_json
                else:
                    invalid_source = active_json
                    config_source = active_json
            except Exception:
                invalid_source = active_json
                config_source = active_json
    if not cfg:
        active = active_yaml_path()
        active_is_symlink = active.is_symlink()
        if active_is_symlink or _safe_path_exists(active):
            resolved = _safe_resolve_path(active) if active_is_symlink else active
            config_source = resolved or active
            cfg_data = try_read_profile_yaml(active)
            if cfg_data:
                cfg = cfg_data
            else:
                invalid_source = config_source
    if not cfg:
        if invalid_source is not None:
            console.print(f"[bold green]{running_backend.display_name} is running[/bold green] (active config unreadable)")
            console.print(f"  [dim]{invalid_source}[/dim]\n")
            return
        console.print(f"[bold green]{running_backend.display_name} is running[/bold green] (no active config found)")
        return

    model_path = cfg.get("model", "?")
    model_name = model_path.split("/")[-1] if "/" in str(model_path) else model_path
    port = cfg.get("port", 8888)

    console.print(f"\n[bold green]{running_backend.display_name} is running[/bold green]")
    console.print(f"  [bold]Model[/bold]      {model_name}")
    console.print(f"  [bold]Endpoint[/bold]   http://localhost:{port}/v1")
    console.print()

    console.print("  [bold]Serving config[/bold]")

    if running_backend.name == "llamacpp":
        ctx = cfg.get("ctx_size", "?")
        ctx_display = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        console.print(f"    Context window:    {ctx_display}")
        console.print(f"    Concurrent slots:  {cfg.get('parallel', '?')}  (max in-flight requests)")
        ngl = cfg.get("n_gpu_layers")
        if ngl is not None:
            console.print(f"    GPU layers:        {ngl}")
        if cfg.get("flash_attn"):
            console.print("    Flash attention:   on")
        if cfg.get("embedding"):
            pooling = cfg.get("pooling", "mean")
            console.print(f"    Mode:              embedding (pooling: {pooling})")
        if cfg.get("jinja"):
            console.print("    Tool calling:      enabled (--jinja)")
    else:
        # vLLM
        ctx = cfg.get("max-model-len", "?")
        ctx_display = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        console.print(f"    Context window:    {ctx_display}")
        console.print(f"    Concurrent slots:  {cfg.get('max-num-seqs', '?')}  (max in-flight requests)")
        console.print(f"    KV cache dtype:    {cfg.get('kv-cache-dtype', 'auto')}")
        if cfg.get("enable-prefix-caching"):
            console.print("    Prefix caching:    enabled")
        bt = cfg.get("max-num-batched-tokens")
        if bt:
            console.print(f"    Batched tokens:    {bt}")
        gpu_util = cfg.get("gpu-memory-utilization")
        if isinstance(gpu_util, float):
            console.print(f"    GPU memory:        {gpu_util:.1%}")

    # GPU memory bar (shared for both backends)
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip().split("\n")[0]
        used, total = [int(x.strip()) for x in out.split(",")]
        pct = used * 100 // total if total else 0
        bar_w = 20
        filled = pct * bar_w // 100
        bar = "[green]" + "█" * filled + "[/green][dim]" + "░" * (bar_w - filled) + "[/dim]"
        console.print(f"\n  [bold]GPU[/bold]        {bar}  {used / 1024:.1f} / {total / 1024:.1f} GB ({pct}%)")
    except Exception:
        pass

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
        _build_config, reset_config, find_systemd_unit_path,
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

    # Discover optional backends before printing backend-specific guidance.
    detected_vllm_root = _discover_vllm_root()
    root = detected_vllm_root or _Path("/opt/vllm")
    llamacpp_root = None
    llamacpp_bin = shutil.which("llama-server")
    llamacpp_candidate = _Path("/opt/llama-cpp")
    if (llamacpp_candidate / "bin" / "llama-server").exists():
        llamacpp_root = llamacpp_candidate
    elif llamacpp_bin:
        llamacpp_root = _Path(llamacpp_bin).resolve().parent.parent

    # ── vLLM ──
    if detected_vllm_root:
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
    elif llamacpp_root:
        _warn("vLLM        not found (optional — needed for safetensors models)")
    else:
        _warn("vLLM        not found (install it for safetensors models)")
        _warn("            See: https://docs.vllm.ai/en/latest/getting_started/installation.html")

    # ── llama.cpp ──
    if llamacpp_root == llamacpp_candidate:
        _ok(f"llama.cpp   found at {llamacpp_candidate}")
    elif llamacpp_bin:
        _ok(f"llama.cpp   {llamacpp_bin}")
    elif detected_vllm_root:
        _warn("llama.cpp   not found (optional — for GGUF models)")
    else:
        _warn("llama.cpp   not found (install it for GGUF models)")

    if not detected_vllm_root and not llamacpp_root:
        _warn("Backends    no serving backend detected yet")
        _warn("            Install vLLM and/or llama.cpp, then rerun vserve init")

    # ── Models ──
    if detected_vllm_root:
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

    # ── systemd (vLLM) ──
    svc_name, svc_user = _discover_service()
    if detected_vllm_root:
        svc_path = find_systemd_unit_path(svc_name)
        if svc_path is not None:
            _ok(f"systemd     {svc_name}.service (user: {svc_user})")
        else:
            _fail("systemd     no vLLM systemd service found (required for vLLM runs)")
            _fail(f"            Create /etc/systemd/system/{svc_name}.service with:")
            _fail("            [Service] ExecStart=/opt/vllm/venv/bin/vllm serve ...")
    else:
        _warn("systemd     vLLM service not checked (vLLM not installed)")

    # ── systemd (llama.cpp) ──
    if llamacpp_root:
        lc_svc_name = "llama-cpp"
        lc_svc_user = "llama-cpp"
        lc_svc_path = find_systemd_unit_path(lc_svc_name)
        if lc_svc_path is not None:
            _ok(f"systemd     {lc_svc_name}.service")
        else:
            _warn(f"systemd     no {lc_svc_name}.service found")
            _warn(f"            Create /etc/systemd/system/{lc_svc_name}.service with:")
            _warn(f"            [Service] ExecStart={llamacpp_root}/configs/active.sh")

        # Service user
        try:
            r = subprocess.run(["id", lc_svc_user], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                _ok(f"user        {lc_svc_user} exists")
            else:
                _warn(f"user        {lc_svc_user} not found")
                _warn(f"            sudo useradd -r -s /usr/sbin/nologin -g llm {lc_svc_user}")
        except Exception:
            pass

    # ── Port ──
    port = _discover_port(root)
    _ok(f"Port        {port}")

    # ── NVML fan control ──
    try:
        from vserve.gpu import get_fan_count

        fan_count = get_fan_count()
        if fan_count > 0:
            _ok(f"Fan control {fan_count} fan(s) exposed via NVML (root required)")
        else:
            _warn("Fan control no controllable fans exposed via NVML")
    except Exception:
        _warn("Fan control NVML unavailable")

    # ── gum (interactive UI) ──
    gum_available = _has_gum()
    if gum_available:
        _ok("gum         installed (enhanced interactive menus)")
    else:
        _warn("gum         not installed (using built-in menus)")

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
    elif not gum_available:
        if has_banner:
            console.print("  [yellow]Login banner is installed but inactive until gum is installed.[/yellow]")
        else:
            console.print("  [dim]Login banner unavailable until gum is installed.[/dim]")
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
    if not detected_vllm_root and not llamacpp_root:
        console.print("  Install vLLM and/or llama.cpp, then rerun vserve init")
    console.print("  vserve add      Search & download a model")
    console.print("  vserve doctor   Full system check")
    console.print("  vserve          Dashboard")
    console.print()


def _doctor_summary_label(warn_count: int, fail_count: int) -> str:
    if fail_count == 0 and warn_count == 0:
        return "All clear"
    if fail_count == 0:
        return f"{warn_count} warning(s) found"
    if warn_count == 0:
        return f"{fail_count} issue(s) found"
    return f"{fail_count} issue(s) and {warn_count} warning(s) found"


@app.command()
def doctor():
    """Check system readiness."""
    import os
    import subprocess
    import socket
    from pathlib import Path

    from vserve.config import (
        VLLM_BIN,
        VLLM_ROOT,
        active_yaml_path,
        try_read_profile_yaml,
        LOGS_DIR,
        find_systemd_unit_path,
    )
    from vserve.backends import any_backend_running

    ok_count = 0
    warn_count = 0
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
        nonlocal warn_count
        console.print(f"  [yellow]WARN[/yellow]  {msg}")
        if fix:
            console.print(f"          Fix: {fix}")
        warn_count += 1

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
        if r.returncode == 0:
            ver = r.stdout.strip() or r.stderr.strip() or "found"
            _ok(f"vLLM {ver}")
        else:
            details = r.stderr.strip() or r.stdout.strip()
            _fail(f"vLLM not working at {VLLM_BIN}", details or "Check the vLLM installation and environment")
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

    # -- Per-backend checks --
    from vserve.config import cfg as _cfg
    _c = _cfg()

    # ── vLLM ──
    console.print("\n  [bold]vLLM[/bold]")

    # Service user
    try:
        r = subprocess.run(["id", _c.service_user], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            _ok(f"Service user '{_c.service_user}' exists")
        else:
            _fail(f"Service user '{_c.service_user}' not found")
    except Exception:
        _fail("Cannot check service user")

    # systemd unit
    svc_path = find_systemd_unit_path(_c.service_name)
    if svc_path is not None:
        try:
            svc_content = svc_path.read_text()
        except (OSError, UnicodeDecodeError) as exc:
            _warn(f"Could not read systemd unit: {svc_path.name}", str(exc))
        else:
            if "ProtectSystem=strict" in svc_content:
                _fail("systemd unit has ProtectSystem=strict",
                      "Remove it — breaks nvcc JIT compilation")
            elif "TimeoutStartSec" not in svc_content:
                _warn("No TimeoutStartSec in service — default 90s may be too short for JIT",
                      "Add TimeoutStartSec=600")
            else:
                _ok("systemd unit configured correctly")
    else:
        _fail(f"No systemd unit found for {_c.service_name}.service",
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
        except (OSError, UnicodeDecodeError) as exc:
            _warn(f".env unreadable: {env_path}", str(exc))
    else:
        _fail(f"No .env at {env_path}")

    # Models directory
    vllm_models = VLLM_ROOT / "models"
    if vllm_models.exists():
        try:
            vllm_mc = sum(1 for p in vllm_models.glob("*/*/config.json"))
        except OSError as exc:
            _warn(f"Models dir unreadable: {vllm_models}", str(exc))
        else:
            _ok(f"Models dir: {vllm_models} ({vllm_mc} models)")
    else:
        _warn(f"Models dir {vllm_models} does not exist")

    # Caches
    cache_checks = [
        (VLLM_ROOT / ".cache" / "flashinfer", "FlashInfer JIT"),
        (VLLM_ROOT / ".cache" / "vllm" / "torch_compile_cache", "torch.compile"),
    ]
    for cdir, label in cache_checks:
        if not cdir.exists():
            _warn(f"{label} cache missing — first start will JIT compile (2-10 min)")
            continue
        try:
            files = []
            for f in cdir.rglob("*"):
                try:
                    if f.is_file():
                        files.append(f)
                except OSError:
                    continue
        except OSError as exc:
            _warn(f"{label} cache unreadable", str(exc))
            continue
        if not files:
            _warn(f"{label} cache dir exists but is empty — first start may be slow")
            continue
        size_bytes = 0
        for f in files:
            try:
                size_bytes += f.stat().st_size
            except OSError:
                continue
        _ok(f"{label} cache ({size_bytes / (1024 * 1024):.0f} MB)")

    # Active config
    active = active_yaml_path()
    active_is_symlink = active.is_symlink()
    if active_is_symlink:
        target = _safe_resolve_path(active)
        if target is None:
            _warn("active.yaml symlink is unreadable or recursive")
        elif _safe_path_exists(target):
            cfg_data = try_read_profile_yaml(active)
            if cfg_data is None:
                _warn(f"active.yaml unreadable: {target.name}")
            else:
                model_path = cfg_data.get("model", "")
                if isinstance(model_path, (str, os.PathLike)) and model_path and Path(model_path).exists():
                    _ok(f"active.yaml → {target.name}")
                else:
                    _fail(f"active.yaml model path missing: {model_path}")
        else:
            _fail(f"active.yaml → broken symlink: {target}")
    elif _safe_path_exists(active):
        _ok("active.yaml exists (not a symlink)")
    else:
        _ok("No active.yaml (clean — will be created on vserve run)")

    # TMPDIR
    tmp_dir = VLLM_ROOT / "tmp"
    if tmp_dir.exists() and tmp_dir.is_dir():
        _ok(f"TMPDIR at {tmp_dir}")
        try:
            sockets = list(tmp_dir.glob("*"))
            stale = [s for s in sockets if s.is_socket()]
        except OSError as exc:
            _warn(f"Could not inspect TMPDIR sockets: {tmp_dir}", str(exc))
        else:
            if len(stale) > 10:
                _warn(f"{len(stale)} stale sockets in {tmp_dir}", f"sudo find {tmp_dir} -type s -delete")
    else:
        _fail(f"TMPDIR {tmp_dir} does not exist", f"sudo mkdir -p {tmp_dir} && sudo chown vllm:llm {tmp_dir}")

    # ── llama.cpp ──
    console.print("\n  [bold]llama.cpp[/bold]")

    lc_root = _c.llamacpp_root
    if lc_root is None:
        _warn("llama.cpp not configured (run vserve init to detect)")
    else:
        _ok(f"Root: {lc_root}")

        # Binary
        lc_bin = lc_root / "bin" / "llama-server"
        if lc_bin.exists():
            try:
                r = subprocess.run([str(lc_bin), "--version"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    ver_line = ""
                    for ln in (r.stdout + r.stderr).splitlines():
                        if "version" in ln.lower() or ln.startswith("b"):
                            ver_line = ln.strip()
                            break
                    _ok(f"llama-server {ver_line}" if ver_line else f"llama-server at {lc_bin}")
                else:
                    details = r.stderr.strip() or r.stdout.strip()
                    _fail(f"llama-server at {lc_bin} is not working",
                          details or "Build or reinstall llama.cpp")
            except Exception:
                _fail(f"llama-server at {lc_bin} is not working")
        elif __import__("shutil").which("llama-server"):
            _ok("llama-server found on PATH")
        else:
            _fail("llama-server not found",
                  "Build: cmake -B build -DGGML_CUDA=ON && cmake --build build -t llama-server")

        # Service user
        lc_user = _c.llamacpp_service_user
        try:
            r = subprocess.run(["id", lc_user], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                _ok(f"Service user '{lc_user}' exists")
            else:
                _fail(f"Service user '{lc_user}' not found",
                      f"sudo useradd -r -s /usr/sbin/nologin -g llm {lc_user}")
        except Exception:
            _fail("Cannot check llama-cpp service user")

        # systemd unit (deep inspection, matching vLLM)
        lc_svc = _c.llamacpp_service_name
        lc_svc_path = find_systemd_unit_path(lc_svc)
        if lc_svc_path is not None:
            try:
                lc_svc_content = lc_svc_path.read_text()
            except (OSError, UnicodeDecodeError) as exc:
                _warn(f"Could not read {lc_svc}.service", str(exc))
            else:
                issues = []
                if "CUDA_VISIBLE_DEVICES" not in lc_svc_content:
                    issues.append("missing CUDA_VISIBLE_DEVICES (may use wrong GPU)")
                if "TimeoutStartSec" not in lc_svc_content:
                    issues.append("no TimeoutStartSec (large models need time)")
                if issues:
                    _warn(f"{lc_svc}.service: {'; '.join(issues)}")
                else:
                    _ok(f"{lc_svc}.service unit configured correctly")
        else:
            _fail(f"No systemd unit found for {lc_svc}.service",
                  f"Create /etc/systemd/system/{lc_svc}.service with ExecStart={lc_root}/configs/active.sh")

        # Models directory
        lc_models = lc_root / "models"
        if lc_models.exists():
            try:
                model_count = sum(1 for p in lc_models.rglob("*.gguf") if p.is_file())
            except OSError as exc:
                _warn(f"Models dir unreadable: {lc_models}", str(exc))
            else:
                _ok(f"Models dir: {lc_models} ({model_count} GGUF files)")
        else:
            _warn(f"Models dir {lc_models} does not exist",
                  f"sudo mkdir -p {lc_models} && sudo chown {lc_user}:llm {lc_models}")

        # Configs directory + active config
        lc_configs = lc_root / "configs"
        if lc_configs.exists():
            active_sh = lc_configs / "active.sh"
            active_json = lc_configs / "active.json"
            if active_sh.exists():
                _ok("active.sh launch script present")
            else:
                _ok(f"Configs dir: {lc_configs} (no active config yet)")
            if active_json.exists():
                try:
                    import json as _json2
                    lc_cfg = _json2.loads(active_json.read_text())
                    lc_model = lc_cfg.get("model", "")
                    if isinstance(lc_model, (str, os.PathLike)) and lc_model and Path(lc_model).exists():
                        _ok(f"active.json → {Path(lc_model).name}")
                    elif lc_model:
                        _fail(f"active.json model path missing: {lc_model}")
                except Exception:
                    _warn("active.json exists but unreadable")
        else:
            _warn(f"Configs dir {lc_configs} does not exist",
                  f"sudo mkdir -p {lc_configs} && sudo chown {lc_user}:llm {lc_configs}")

        # gguf package
        try:
            import gguf  # type: ignore[import-untyped]  # noqa: F401
            _ok("gguf package installed")
        except ImportError:
            _warn("gguf package not installed — needed for GGUF metadata reading",
                  "pip install 'vserve\\[llamacpp]'")

    # -- Shared --
    console.print("\n  [bold]Shared[/bold]")

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
        try:
            size_mb = log_file.stat().st_size / (1024 * 1024)
        except OSError as exc:
            _warn(f"Could not read log metadata: {log_file}", str(exc))
        else:
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

    console.print(
        f"\n  [bold]{_doctor_summary_label(warn_count, fail_count)}[/bold]  "
        f"({ok_count} ok, {warn_count} warn, {fail_count} fail)\n",
    )


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
        console.print("  [yellow]Next vserve run will recompile kernels (~5-10 min)[/yellow]")
