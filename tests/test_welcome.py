from pathlib import Path
import os
import subprocess


def _render_welcome(tmp_path, *, nvidia: bool = True, active: bool = False):
    fake_bin = tmp_path / "bin"
    fake_home = tmp_path / "home"
    vllm_root = tmp_path / "vllm"
    llama_root = tmp_path / "llama-cpp"
    config_dir = fake_home / ".config" / "vserve"
    log_path = tmp_path / "gum.log"
    fake_bin.mkdir()
    config_dir.mkdir(parents=True)
    (vllm_root / "models" / "org" / "model-a").mkdir(parents=True)
    (llama_root / "models" / "org" / "model-b").mkdir(parents=True)
    (vllm_root / "configs").mkdir(parents=True)
    (vllm_root / "configs" / "active.yaml").write_text(f"model: {vllm_root}/models/org/model-a\n")
    config_dir.joinpath("config.yaml").write_text(
        "# vserve configuration\n"
        "schema_version: 2\n"
        "cuda_home: /usr/local/cuda\n"
        "backends:\n"
        "  vllm:\n"
        f"    root: {vllm_root}\n"
        "    service_name: vllm\n"
        "    service_user: vllm\n"
        "    port: 9999\n"
        "  llamacpp:\n"
        f"    root: {llama_root}\n"
        "    service_name: llama-cpp\n"
        "    service_user: llama-cpp\n"
    )

    fake_bin.joinpath("gum").write_text(
        "#!/bin/sh\n"
        "printf 'call\\n' >> \"$VSERVE_TEST_GUM_LOG\"\n"
        "shift\n"
        "printf '%s\\n' \"$*\"\n"
    )
    if nvidia:
        fake_bin.joinpath("nvidia-smi").write_text(
            "#!/bin/sh\n"
            "case \"$*\" in\n"
            "  *driver_version*) printf '555.55\\n' ;;\n"
            "  *query-compute-apps*) printf '1234, 2048, /opt/vllm/bin/python\\n' ;;\n"
            "  *query-gpu*) printf '12, 1024, 24576, Test GPU\\n' ;;\n"
            "  *) printf '| NVIDIA-SMI 555.55 CUDA Version: 12.8 |\\n' ;;\n"
            "esac\n"
        )
    else:
        fake_bin.joinpath("nvidia-smi").write_text("#!/bin/sh\nexit 1\n")
    systemctl_exit = "0" if active else "3"
    fake_bin.joinpath("systemctl").write_text(f"#!/bin/sh\nexit {systemctl_exit}\n")
    fake_bin.joinpath("ps").write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *user=*) printf 'vllm\\n' ;;\n"
        "  *etime=*) printf '01:23\\n' ;;\n"
        "esac\n"
    )
    for path in fake_bin.iterdir():
        path.chmod(0o755)

    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "VSERVE_TEST_GUM_LOG": str(log_path),
    }
    result = subprocess.run(
        ["bash", "-lc", "source src/vserve/welcome.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return result, log_path


def test_welcome_script_reads_schema_v2_config():
    script = Path("src/vserve/welcome.sh").read_text()

    assert "yaml.safe_load" in script
    assert "backends" in script
    assert "llamacpp" in script


def test_welcome_script_keeps_renderer_process_count_low(tmp_path):
    result, log_path = _render_welcome(tmp_path)

    assert result.returncode == 0, result.stderr
    gum_calls = log_path.read_text().splitlines()
    assert len(gum_calls) <= 12


def test_welcome_script_renders_rich_dashboard_sections(tmp_path):
    result, _log_path = _render_welcome(tmp_path, active=True)

    assert result.returncode == 0, result.stderr
    assert "vserve" in result.stdout
    assert "CONTROL DECK" in result.stdout
    assert "SERVING" in result.stdout
    assert "GPU TELEMETRY" in result.stdout
    assert "GPU WORKLOAD" in result.stdout
    assert "COMMAND DECK" in result.stdout
    assert "ONLINE" in result.stdout
    assert "http://localhost:9999/v1" in result.stdout
    assert "model-a" in result.stdout
    assert "Test GPU" in result.stdout
    assert "vserve run" in result.stdout


def test_welcome_script_renders_gpu_unavailable_state(tmp_path):
    result, _log_path = _render_welcome(tmp_path, nvidia=False)

    assert result.returncode == 0, result.stderr
    assert "GPU TELEMETRY" in result.stdout
    assert "GPU unavailable" in result.stdout
    assert "No GPU processes" in result.stdout
