import json
import pytest
from vx.tune import generate_sweep_params, pick_winner, PROFILES


def test_profiles_exist():
    assert set(PROFILES.keys()) == {"interactive", "batch", "agentic", "creative"}


def test_generate_interactive_params(fake_limits, tmp_path):
    serve, bench, subcommand = generate_sweep_params(
        profile="interactive", limits=fake_limits, output_dir=tmp_path,
    )
    serve_data = json.loads(serve.read_text())
    bench_data = json.loads(bench.read_text())
    assert len(serve_data) > 0
    assert len(bench_data) > 0
    assert subcommand == "serve"
    for entry in serve_data.values():
        assert "kv_cache_dtype" in entry
    for entry in bench_data.values():
        assert entry["num_prompts"] == 20


def test_generate_batch_params(fake_limits, tmp_path):
    serve, bench, subcommand = generate_sweep_params(
        profile="batch", limits=fake_limits, output_dir=tmp_path,
    )
    assert subcommand == "serve_workload"
    serve_data = json.loads(serve.read_text())
    seqs_values = {e["max_num_seqs"] for e in serve_data.values()}
    assert seqs_values.issubset({8, 16, 32, 64})


def test_generate_agentic_params(fake_limits, tmp_path):
    serve, bench, subcommand = generate_sweep_params(
        profile="agentic", limits=fake_limits, output_dir=tmp_path,
    )
    serve_data = json.loads(serve.read_text())
    for entry in serve_data.values():
        assert entry["max_model_len"] == 65536  # max where fp8 has slots in fake_limits


def test_generate_creative_params(fake_limits, tmp_path):
    serve, bench, subcommand = generate_sweep_params(
        profile="creative", limits=fake_limits, output_dir=tmp_path,
    )
    bench_data = json.loads(bench.read_text())
    for entry in bench_data.values():
        assert entry["random_output_len"] >= 1024


def test_generate_batch_moe_caps_seqs(fake_limits, tmp_path):
    fake_limits["is_moe"] = True
    serve, bench, subcommand = generate_sweep_params(
        profile="batch", limits=fake_limits, output_dir=tmp_path,
    )
    serve_data = json.loads(serve.read_text())
    for entry in serve_data.values():
        assert entry["max_num_seqs"] <= 32


def test_pick_winner_interactive(tmp_path):
    combo = tmp_path / "SERVE--fast-BENCH--b"
    combo.mkdir(parents=True)
    (combo / "summary.json").write_text(json.dumps([
        {"median_tpot_ms": 11.0, "mean_ttft_ms": 40.0, "output_throughput": 85.0, "failed": 0, "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "max_model_len": 4096},
        {"median_tpot_ms": 12.0, "mean_ttft_ms": 42.0, "output_throughput": 83.0, "failed": 0, "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "max_model_len": 4096},
    ]))
    combo2 = tmp_path / "SERVE--slow-BENCH--b"
    combo2.mkdir(parents=True)
    (combo2 / "summary.json").write_text(json.dumps([
        {"median_tpot_ms": 20.0, "mean_ttft_ms": 80.0, "output_throughput": 50.0, "failed": 0, "gpu_memory_utilization": 0.90, "kv_cache_dtype": "auto", "max_model_len": 4096},
    ]))

    winner, top3 = pick_winner(tmp_path, profile="interactive")
    assert winner["params"]["kv_cache_dtype"] == "fp8"


def test_pick_winner_batch(tmp_path):
    combo = tmp_path / "SERVE--high-BENCH--b"
    combo.mkdir(parents=True)
    (combo / "summary.json").write_text(json.dumps([
        {"median_tpot_ms": 8.0, "mean_ttft_ms": 30.0, "output_throughput": 120.0, "failed": 0, "max_num_seqs": 32},
    ]))
    combo2 = tmp_path / "SERVE--low-BENCH--b"
    combo2.mkdir(parents=True)
    (combo2 / "summary.json").write_text(json.dumps([
        {"median_tpot_ms": 12.0, "mean_ttft_ms": 25.0, "output_throughput": 85.0, "failed": 0, "max_num_seqs": 8},
    ]))

    winner, _ = pick_winner(tmp_path, profile="batch")
    assert winner["output_throughput"] == 120.0


def test_pick_winner_no_results(tmp_path):
    with pytest.raises(SystemExit):
        pick_winner(tmp_path, profile="interactive")
