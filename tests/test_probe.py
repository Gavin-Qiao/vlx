import pytest
from vlx.models import detect_model, ModelInfo
from vlx.probe import (
    probe_context_limit,
    probe_concurrency_limit,
    probe_model,
    CONTEXT_STEPS,
)


@pytest.fixture
def sample_model_info(fake_model_dir) -> ModelInfo:
    return detect_model(fake_model_dir)


def test_context_steps_are_powers_of_two():
    for step in CONTEXT_STEPS:
        assert step & (step - 1) == 0, f"{step} is not a power of 2"


def test_probe_context_limit_success(mocker, sample_model_info):
    mocker.patch("vlx.probe._try_start_vllm", return_value=True)
    result = probe_context_limit(
        model_info=sample_model_info,
        context_len=4096,
        kv_dtype="fp8",
        gpu_mem_util=0.958,
    )
    assert result is True


def test_probe_context_limit_oom(mocker, sample_model_info):
    mocker.patch("vlx.probe._try_start_vllm", return_value=False)
    result = probe_context_limit(
        model_info=sample_model_info,
        context_len=262144,
        kv_dtype="auto",
        gpu_mem_util=0.958,
    )
    assert result is False


def test_probe_concurrency_limit(mocker, sample_model_info):
    def fake_start(*, model_info, context_len, kv_dtype, gpu_mem_util, max_num_seqs):
        return max_num_seqs <= 32

    mocker.patch("vlx.probe._try_start_vllm", side_effect=fake_start)
    max_seqs = probe_concurrency_limit(
        model_info=sample_model_info,
        context_len=8192,
        kv_dtype="fp8",
        gpu_mem_util=0.958,
    )
    assert max_seqs == 32


def test_probe_concurrency_limit_none(mocker, sample_model_info):
    mocker.patch("vlx.probe._try_start_vllm", return_value=False)
    max_seqs = probe_concurrency_limit(
        model_info=sample_model_info,
        context_len=262144,
        kv_dtype="auto",
        gpu_mem_util=0.958,
    )
    assert max_seqs is None


def test_probe_model_full(mocker, sample_model_info):
    def fake_start(*, model_info, context_len, kv_dtype, gpu_mem_util, max_num_seqs=1):
        if kv_dtype == "auto" and context_len > 32768:
            return False
        if kv_dtype == "fp8" and context_len > 65536:
            return False
        if max_num_seqs > 16:
            return False
        return True

    mocker.patch("vlx.probe._try_start_vllm", side_effect=fake_start)

    limits = probe_model(
        model_info=sample_model_info,
        gpu_mem_util=0.958,
        vram_total_gb=47.8,
    )

    assert limits["limits"]["4096"]["auto"] == 16
    assert limits["limits"]["4096"]["fp8"] == 16
    assert limits["limits"]["65536"]["auto"] is None
    assert limits["limits"]["65536"]["fp8"] == 16
    assert limits["limits"]["131072"]["fp8"] is None
