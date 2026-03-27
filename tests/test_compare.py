import pytest
from vx.compare import filter_models_for_workload, rank_models


@pytest.fixture
def two_models_limits():
    model_a = {
        "model_path": "/opt/vllm/models/prov/ModelA",
        "limits": {
            "4096": {"auto": 64, "fp8": 128},
            "8192": {"auto": 32, "fp8": 64},
            "16384": {"auto": None, "fp8": 16},
        },
    }
    model_b = {
        "model_path": "/opt/vllm/models/prov/ModelB",
        "limits": {
            "4096": {"auto": 32, "fp8": 64},
            "8192": {"auto": 8, "fp8": 16},
            "16384": {"auto": None, "fp8": None},
        },
    }
    return [model_a, model_b]


def test_filter_models_fits(two_models_limits):
    result = filter_models_for_workload(
        all_limits=two_models_limits, users=10, context=8192,
    )
    assert len(result) == 2


def test_filter_models_excludes(two_models_limits):
    result = filter_models_for_workload(
        all_limits=two_models_limits, users=50, context=8192,
    )
    assert len(result) == 1
    assert result[0]["model_path"].endswith("ModelA")


def test_filter_models_context_too_high(two_models_limits):
    result = filter_models_for_workload(
        all_limits=two_models_limits, users=1, context=32768,
    )
    assert len(result) == 0


def test_rank_models(two_models_limits):
    fits = filter_models_for_workload(
        all_limits=two_models_limits, users=5, context=8192,
    )
    ranked = rank_models(fits)
    assert ranked[0]["model_path"].endswith("ModelA")
