"""Tests for vserve tune CLI command — analytical limit calculation."""

from typer.testing import CliRunner
from vserve.cli import app

runner = CliRunner()


def test_tune_no_model_found(mocker):
    mocker.patch("vserve.cli._all_models", return_value=[])
    mocker.patch("vserve.cli.scan_models", return_value=[])
    result = runner.invoke(app, ["tune", "nonexistent"])
    assert result.exit_code == 1
    assert "No model" in result.output


def _mock_vllm_backend(mocker):
    """Create a mock vLLM backend for tune tests."""
    from unittest.mock import Mock
    from vserve.probe import calculate_limits
    backend = Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.tune.side_effect = lambda m, gpu, **kw: calculate_limits(
        model_info=m, vram_total_gb=gpu.vram_total_gb, gpu_mem_util=kw.get("gpu_mem_util", 0.90),
    )
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    return backend


def test_tune_calculates_limits(mocker, fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    _mock_vllm_backend(mocker)
    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    mocker.patch("vserve.config.write_limits")

    result = runner.invoke(app, ["tune", "TestModel"])
    assert result.exit_code == 0
    assert "slots" in result.output


def test_tune_all_models(mocker, fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    _mock_vllm_backend(mocker)
    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    mocker.patch("vserve.config.write_limits")

    result = runner.invoke(app, ["tune", "--all"])
    assert result.exit_code == 0


def test_tune_cached_shows_table(mocker, fake_model_dir, fake_limits):
    from vserve.models import detect_model
    from vserve.config import LIMITS_SCHEMA_VERSION
    from vserve.runtime import build_tuning_fingerprint
    m = detect_model(fake_model_dir)
    gpu = type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })()
    cached_limits = dict(fake_limits)
    cached_limits["schema_version"] = LIMITS_SCHEMA_VERSION
    cached_limits["backend"] = "vllm"
    cached_limits["fingerprint"] = build_tuning_fingerprint(
        model_info=m,
        gpu=gpu,
        backend="vllm",
        gpu_mem_util=fake_limits["gpu_memory_utilization"],
    )

    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.cli.read_limits", return_value=cached_limits)
    mocker.patch("vserve.gpu.get_gpu_info", return_value=gpu)
    # Mock config to not have custom gpu settings (so gpu_util matches fake_limits)
    mock_cfg = mocker.MagicMock()
    mock_cfg.gpu_memory_utilization = fake_limits["gpu_memory_utilization"]
    mock_cfg.gpu_overhead_gb = None
    mocker.patch("vserve.config.cfg", return_value=mock_cfg)

    result = runner.invoke(app, ["tune", "TestModel"])
    assert result.exit_code == 0
    assert "cached" in result.output


def test_tune_recalculates_legacy_limits_without_schema(mocker, fake_model_dir, fake_limits):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    _mock_vllm_backend(mocker)
    legacy_limits = dict(fake_limits)
    legacy_limits.pop("schema_version", None)
    legacy_limits.pop("fingerprint", None)
    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.cli.read_limits", return_value=legacy_limits)
    write_limits = mocker.patch("vserve.config.write_limits")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    mock_cfg = mocker.MagicMock()
    mock_cfg.gpu_memory_utilization = fake_limits["gpu_memory_utilization"]
    mock_cfg.gpu_overhead_gb = None
    mocker.patch("vserve.config.cfg", return_value=mock_cfg)

    result = runner.invoke(app, ["tune", "TestModel"])

    assert result.exit_code == 0
    assert "cached" not in result.output
    write_limits.assert_called_once()


def test_tune_recalc_ignores_cache(mocker, fake_model_dir, fake_limits):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    _mock_vllm_backend(mocker)
    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.cli.read_limits", return_value=fake_limits)
    mocker.patch("vserve.config.write_limits")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "TestModel", "--recalc"])
    assert result.exit_code == 0
    assert "cached" not in result.output
    assert "slots" in result.output


def test_tune_bench_saves_bounded_benchmark_result(mocker, fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    backend = _mock_vllm_backend(mocker)
    backend.is_running.return_value = False
    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    write_limits = mocker.patch("vserve.config.write_limits")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    bench = mocker.patch(
        "vserve.cli._run_tuning_benchmarks",
        return_value={"status": "ok", "results": [{"profile": "balanced"}]},
    )

    result = runner.invoke(app, ["tune", "TestModel", "--bench", "--bench-seconds", "45", "--bench-candidates", "1"])

    assert result.exit_code == 0
    bench.assert_called_once()
    assert bench.call_args.kwargs["bench_seconds"] == 45
    assert bench.call_args.kwargs["bench_startup_seconds"] == 300
    saved = write_limits.call_args.args[1]
    assert saved["benchmark_results"]["status"] == "ok"


def test_tune_all_with_bench_runs_bench_for_each_model(mocker, fake_model_dir, fake_moe_model_dir):
    from vserve.models import detect_model
    models = [detect_model(fake_model_dir), detect_model(fake_moe_model_dir)]

    _mock_vllm_backend(mocker)
    mocker.patch("vserve.cli._all_models", return_value=models)
    mocker.patch("vserve.cli.scan_models", return_value=models)
    mocker.patch("vserve.config.write_limits")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    bench = mocker.patch(
        "vserve.cli._run_tuning_benchmarks",
        return_value={"status": "ok", "results": [{"profile": "balanced"}]},
    )

    result = runner.invoke(app, ["tune", "--all", "--bench", "--bench-candidates", "1", "--bench-seconds", "30", "--bench-startup-seconds", "90"])

    assert result.exit_code == 0
    assert bench.call_count == 2
    assert bench.call_args.kwargs["bench_startup_seconds"] == 90


def test_benchmark_candidates_start_with_interactivity():
    from vserve.cli import _benchmark_candidate_names

    result = _benchmark_candidate_names(
        {
            "recommendations": {
                "balanced": {},
                "interactivity": {},
                "throughput": {},
            }
        },
        max_candidates=2,
    )

    assert result == ["interactivity", "balanced"]


def test_benchmark_passes_separate_startup_timeout(mocker, fake_model_dir, tmp_path):
    from vserve.cli import _run_tuning_benchmarks
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.is_running.return_value = False
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "bench.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    launch = mocker.patch("vserve.cli._launch_backend")
    mocker.patch("vserve.bench.run_openai_completion_benchmark", return_value={"status": "ok", "requests_completed": 1})

    result = _run_tuning_benchmarks(
        model,
        backend,
        {
            "recommendations": {
                "interactivity": {
                    "context": 4096,
                    "kv_cache_dtype": "auto",
                    "max_num_seqs": 2,
                    "max_num_batched_tokens": 4096,
                },
            },
        },
        gpu_mem_util=0.9,
        bench_seconds=30,
        bench_startup_seconds=180,
        bench_candidates=1,
        bench_requests=1,
    )

    assert result["status"] == "ok"
    assert launch.call_args.kwargs["health_timeout_s"] == 180


def test_benchmark_records_actual_measurement_timeout(mocker, fake_model_dir, tmp_path):
    from vserve.cli import _run_tuning_benchmarks
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.is_running.return_value = False
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "bench.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    mocker.patch("vserve.cli._launch_backend")
    measurement = mocker.patch(
        "vserve.bench.run_openai_completion_benchmark",
        return_value={"status": "ok", "requests_completed": 1},
    )

    result = _run_tuning_benchmarks(
        model,
        backend,
        {
            "recommendations": {
                "balanced": {
                    "context": 4096,
                    "kv_cache_dtype": "auto",
                    "max_num_seqs": 2,
                    "max_num_batched_tokens": 4096,
                },
            },
        },
        gpu_mem_util=0.9,
        bench_seconds=120,
        bench_startup_seconds=180,
        bench_candidates=1,
        bench_requests=1,
    )

    assert result["status"] == "ok"
    assert result["measurement_timeout_seconds"] == 45.0
    assert measurement.call_args.kwargs["timeout_s"] == 45.0


def test_benchmark_stops_backend_when_launch_times_out(mocker, fake_model_dir, tmp_path):
    from vserve.cli import _run_tuning_benchmarks
    from vserve.models import detect_model

    model = detect_model(fake_model_dir)
    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.is_running.return_value = False
    backend.build_config.return_value = {"model": str(model.path), "port": 8888}
    mocker.patch("vserve.config.profile_path", return_value=tmp_path / "bench.yaml")
    mocker.patch("vserve.config.write_profile_yaml")
    mocker.patch("vserve.cli._launch_backend", side_effect=RuntimeError("health timeout"))
    clear_session = mocker.patch("vserve.cli.clear_session")

    result = _run_tuning_benchmarks(
        model,
        backend,
        {
            "recommendations": {
                "balanced": {
                    "context": 4096,
                    "kv_cache_dtype": "auto",
                    "max_num_seqs": 2,
                    "max_num_batched_tokens": 4096,
                },
            },
        },
        gpu_mem_util=0.9,
        bench_seconds=30,
        bench_candidates=1,
        bench_requests=1,
    )

    assert result["status"] == "startup_timeout"
    backend.stop.assert_called_once_with(non_interactive=True)
    clear_session.assert_called_once()


def test_benchmark_llamacpp_runs_openai_completion_measurement(mocker, fake_gguf_model_dir, tmp_path):
    from vserve.cli import _run_tuning_benchmarks
    from vserve.models import detect_model

    model = detect_model(fake_gguf_model_dir)
    backend = mocker.Mock()
    backend.name = "llamacpp"
    backend.display_name = "llama.cpp"
    backend.root_dir = tmp_path
    backend.is_running.return_value = False
    backend.build_config.return_value = {"model": str(model.path / "model.gguf"), "port": 8888}
    launch = mocker.patch("vserve.cli._launch_backend")
    measurement = mocker.patch(
        "vserve.bench.run_openai_completion_benchmark",
        return_value={"status": "ok", "requests_completed": 1, "tokens_per_second": 12.5},
    )

    result = _run_tuning_benchmarks(
        model,
        backend,
        {
            "limits": {"4096": 4},
            "n_gpu_layers": 32,
            "pooling": None,
        },
        gpu_mem_util=0.9,
        bench_seconds=30,
        bench_candidates=1,
        bench_requests=1,
        bench_startup_seconds=180,
    )

    assert result["status"] == "ok"
    measurement.assert_called_once()
    cfg_path = launch.call_args.args[1]
    assert cfg_path.suffix == ".json"
    assert cfg_path.exists()
    choices = backend.build_config.call_args.args[1]
    assert choices["context"] == 4096
    assert choices["parallel"] == 4
    assert choices["n_gpu_layers"] == 32


def test_benchmark_llamacpp_embedding_uses_embeddings_endpoint(mocker, fake_embedding_model_dir, tmp_path):
    from vserve.cli import _run_tuning_benchmarks
    from vserve.models import detect_model

    model = detect_model(fake_embedding_model_dir)
    backend = mocker.Mock()
    backend.name = "llamacpp"
    backend.display_name = "llama.cpp"
    backend.root_dir = tmp_path
    backend.is_running.return_value = False
    backend.build_config.return_value = {"model": str(model.path / "model.gguf"), "port": 8888}
    mocker.patch("vserve.cli._launch_backend")
    completion = mocker.patch("vserve.bench.run_openai_completion_benchmark")
    embedding = mocker.patch(
        "vserve.bench.run_openai_embedding_benchmark",
        return_value={"status": "ok", "requests_completed": 1, "items_per_second": 9.0},
    )

    result = _run_tuning_benchmarks(
        model,
        backend,
        {
            "limits": {"4096": 4},
            "n_gpu_layers": 32,
            "is_embedding": True,
            "pooling": "mean",
        },
        gpu_mem_util=0.9,
        bench_seconds=30,
        bench_candidates=1,
        bench_requests=1,
        bench_startup_seconds=180,
    )

    assert result["status"] == "ok"
    embedding.assert_called_once()
    completion.assert_not_called()
    choices = backend.build_config.call_args.args[1]
    assert choices["embedding"] is True
    assert choices["pooling"] == "mean"


def test_print_limits_table_shows_llamacpp_benchmark_summary(mocker, fake_gguf_model_dir):
    from io import StringIO
    from rich.console import Console
    import vserve.cli as cli
    from vserve.models import detect_model

    output = StringIO()
    mocker.patch.object(cli, "console", Console(file=output, force_terminal=False, width=120))

    cli._print_limits_table(
        {
            "limits": {"4096": 4},
            "n_gpu_layers": 32,
            "num_layers": 32,
            "benchmark_results": {
                "status": "ok",
                "results": [
                    {
                        "profile": "interactive",
                        "measurement": {
                            "status": "ok",
                            "tokens_per_second": 11.5,
                            "p95_latency_ms": 220.0,
                        },
                    }
                ],
            },
        },
        detect_model(fake_gguf_model_dir),
    )

    text = output.getvalue()
    assert "Benchmark: ok" in text
    assert "interactive: 11.5 tok/s, p95 220.0 ms" in text


def test_tune_gpu_util_out_of_range(mocker, fake_model_dir):
    """gpu_util outside 0.5-0.99 should fail with a clear error."""
    result = runner.invoke(app, ["tune", "TestModel", "--gpu-util", "0.3"])
    assert result.exit_code == 1
    assert "--gpu-util must be between 0.5 and 0.99" in result.output

    result = runner.invoke(app, ["tune", "TestModel", "--gpu-util", "1.0"])
    assert result.exit_code == 1
    assert "--gpu-util must be between 0.5 and 0.99" in result.output


def test_tune_missing_arch_fields(mocker, tmp_path):
    """Model without architecture fields shows clear error, doesn't crash."""
    import json
    from vserve.models import detect_model
    model_dir = tmp_path / "models" / "test" / "Minimal"
    model_dir.mkdir(parents=True)
    config = {"architectures": ["MinimalLM"], "model_type": "test"}
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)
    m = detect_model(model_dir)

    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.cli.read_limits", return_value=None)
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "Minimal"])
    assert result.exit_code == 0  # continues past bad model
    assert "Missing architecture" in result.output
