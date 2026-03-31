"""Tests for vserve.fan — piecewise-linear fan curve with quiet hours."""

from vserve.fan import (
    compute_fan_speed,
    read_state,
    _interpolate_curve,
    EMERGENCY_TEMP,
    MIN_FAN,
    HYSTERESIS,
    MAX_RAMP_PER_CYCLE,
    MAX_CONSECUTIVE_FAILURES,
    FAILURE_RESET_THRESHOLD,
    _CURVE,
)


class TestCurveShape:
    """Verify the piecewise-linear curve behaves correctly."""

    def test_below_curve_start(self):
        assert _interpolate_curve(20, 100) == MIN_FAN

    def test_at_curve_start(self):
        assert _interpolate_curve(35, 100) == MIN_FAN

    def test_above_curve_end(self):
        assert _interpolate_curve(90, 100) == 100

    def test_at_curve_end(self):
        assert _interpolate_curve(80, 100) == 100

    def test_midpoint_first_segment(self):
        # 35→30%, 55→35%. At 45C: 30 + 0.5 * 5 = 32
        speed = _interpolate_curve(45, 100)
        assert 30 <= speed <= 35

    def test_steep_segment(self):
        # 75→75%, 80→100%. At 78C: 75 + 0.6 * 25 = 90
        speed = _interpolate_curve(78, 100)
        assert 85 <= speed <= 95

    def test_curve_is_monotonic(self):
        """Fan speed never decreases as temperature rises."""
        prev = 0
        for temp in range(0, 100):
            speed = _interpolate_curve(temp, 100)
            assert speed >= prev, f"Non-monotonic at {temp}C: {speed} < {prev}"
            prev = speed

    def test_cap_limits_output(self):
        """During quiet hours (cap=60), curve never exceeds 60."""
        for temp in range(0, EMERGENCY_TEMP):
            speed = _interpolate_curve(temp, 60)
            assert speed <= 60, f"Exceeded cap at {temp}C: {speed}"

    def test_cap_scales_smoothly(self):
        """With cap=60, the curve still ramps — no hard cliff."""
        speed_65 = _interpolate_curve(65, 60)
        speed_75 = _interpolate_curve(75, 60)
        assert speed_75 > speed_65, "Curve should still ramp with cap"


class TestComputeFanSpeed:
    """Temperature -> fan speed with quiet-hours and emergency override."""

    # --- Cold GPU: always minimum ---

    def test_cold_gpu_daytime(self):
        assert compute_fan_speed(temp=30, hour=12) == MIN_FAN

    def test_cold_gpu_nighttime(self):
        assert compute_fan_speed(temp=30, hour=22) == MIN_FAN

    # --- Hot GPU: hits cap ---

    def test_hot_gpu_daytime_capped(self):
        speed = compute_fan_speed(temp=85, hour=12)
        assert speed <= 60

    def test_hot_gpu_nighttime_full(self):
        assert compute_fan_speed(temp=85, hour=22) == 100

    # --- At boundary temps ---

    def test_at_curve_start(self):
        assert compute_fan_speed(temp=35, hour=12) == MIN_FAN

    def test_at_curve_end_daytime(self):
        speed = compute_fan_speed(temp=80, hour=12)
        assert speed <= 60  # capped by quiet hours

    def test_at_curve_end_nighttime(self):
        assert compute_fan_speed(temp=80, hour=22) == 100

    # --- Quiet-hours boundaries ---

    def test_hour_9_is_quiet(self):
        speed = compute_fan_speed(temp=85, hour=9)
        assert speed <= 60

    def test_hour_17_is_quiet(self):
        speed = compute_fan_speed(temp=85, hour=17)
        assert speed <= 60

    def test_hour_18_is_loud(self):
        assert compute_fan_speed(temp=85, hour=18) == 100

    def test_hour_8_is_loud(self):
        assert compute_fan_speed(temp=85, hour=8) == 100

    # --- Custom quiet hours ---

    def test_custom_quiet_hours(self):
        speed = compute_fan_speed(temp=85, hour=23, quiet_start=22, quiet_end=6, quiet_max=50)
        assert speed <= 50

    def test_custom_quiet_max(self):
        speed = compute_fan_speed(temp=85, hour=12, quiet_max=70)
        assert speed <= 70

    # --- Emergency override ---

    def test_emergency_override_daytime(self):
        assert compute_fan_speed(temp=EMERGENCY_TEMP, hour=12) == 100

    def test_emergency_override_nighttime(self):
        assert compute_fan_speed(temp=EMERGENCY_TEMP, hour=22) == 100

    def test_just_below_emergency_daytime_capped(self):
        speed = compute_fan_speed(temp=EMERGENCY_TEMP - 1, hour=12)
        assert speed <= 60

    def test_emergency_with_custom_quiet(self):
        assert compute_fan_speed(temp=90, hour=12, quiet_max=40) == 100

    # --- Never below MIN_FAN, never above 100 ---

    def test_never_below_minimum(self):
        for temp in range(0, 100):
            for hour in range(24):
                speed = compute_fan_speed(temp=temp, hour=hour)
                assert speed >= MIN_FAN
                assert speed <= 100

    # --- Quiet curve is smoother than a hard cap ---

    def test_quiet_curve_no_cliff(self):
        """Fan speed changes gradually during quiet hours — no sudden jumps."""
        speeds = [compute_fan_speed(temp=t, hour=12) for t in range(35, EMERGENCY_TEMP)]
        for i in range(1, len(speeds)):
            delta = speeds[i] - speeds[i - 1]
            assert delta >= 0, f"Non-monotonic at {35 + i}C"
            assert delta <= 10, f"Jump too large at {35 + i}C: {delta}%"


class TestReadState:
    def _patch_paths(self, mocker, tmp_path):
        import vserve.fan
        vserve.fan._paths_resolved = False
        mocker.patch("vserve.fan.cfg", return_value=mocker.Mock(
            logs_dir=tmp_path, run_dir=tmp_path,
        ))

    def test_missing_file(self, tmp_path, mocker):
        self._patch_paths(mocker, tmp_path)
        assert read_state() is None

    def test_valid_json(self, tmp_path, mocker):
        self._patch_paths(mocker, tmp_path)
        f = tmp_path / "vserve-fan.json"
        f.write_text('{"quiet_start": 22, "quiet_end": 6, "quiet_max": 50}')
        state = read_state()
        assert state == {"quiet_start": 22, "quiet_end": 6, "quiet_max": 50}

    def test_corrupted_json(self, tmp_path, mocker):
        self._patch_paths(mocker, tmp_path)
        f = tmp_path / "vserve-fan.json"
        f.write_text("not json{{{")
        assert read_state() is None


class TestConstants:
    def test_failure_threshold_ordering(self):
        assert FAILURE_RESET_THRESHOLD < MAX_CONSECUTIVE_FAILURES

    def test_max_failures_is_reasonable(self):
        assert 20 <= MAX_CONSECUTIVE_FAILURES <= 60

    def test_hysteresis_reasonable(self):
        assert 2 <= HYSTERESIS <= 8

    def test_ramp_limit_reasonable(self):
        assert 3 <= MAX_RAMP_PER_CYCLE <= 10

    def test_curve_is_sorted(self):
        temps = [t for t, _ in _CURVE]
        assert temps == sorted(temps)

    def test_curve_speeds_increase(self):
        speeds = [s for _, s in _CURVE]
        assert speeds == sorted(speeds)

    def test_curve_starts_at_min_fan(self):
        assert _CURVE[0][1] == MIN_FAN

    def test_curve_ends_at_100(self):
        assert _CURVE[-1][1] == 100


class TestResolvePaths:
    def test_resolve_sets_paths(self, tmp_path, mocker):
        import vserve.fan
        vserve.fan._paths_resolved = False
        mocker.patch("vserve.fan.cfg", return_value=mocker.Mock(
            logs_dir=tmp_path / "logs",
            run_dir=tmp_path / "run",
        ))
        vserve.fan._resolve_paths()
        assert vserve.fan.PID_PATH == tmp_path / "run" / "vserve-fan.pid"
        assert vserve.fan.STATE_PATH == tmp_path / "run" / "vserve-fan.json"
        assert vserve.fan.LOG_PATH == tmp_path / "logs" / "vserve-fan.log"

    def test_resolve_is_idempotent(self, tmp_path, mocker):
        import vserve.fan
        vserve.fan._paths_resolved = False
        mocker.patch("vserve.fan.cfg", return_value=mocker.Mock(
            logs_dir=tmp_path, run_dir=tmp_path,
        ))
        vserve.fan._resolve_paths()
        first = vserve.fan.PID_PATH
        vserve.fan._resolve_paths()
        assert vserve.fan.PID_PATH is first

    def test_cli_import_pattern_works(self, tmp_path, mocker):
        import vserve.fan as _fan
        _fan._paths_resolved = False
        mocker.patch("vserve.fan.cfg", return_value=mocker.Mock(
            logs_dir=tmp_path, run_dir=tmp_path,
        ))
        _fan._resolve_paths()
        PID_PATH = _fan.PID_PATH
        STATE_PATH = _fan.STATE_PATH
        from pathlib import Path
        assert isinstance(PID_PATH, Path)
        assert isinstance(STATE_PATH, Path)


def test_run_fixed_daemon_exists():
    """run_fixed_daemon is importable and callable."""
    from vserve.fan import run_fixed_daemon
    assert callable(run_fixed_daemon)


def test_run_daemon_exists():
    """run_daemon is importable and callable."""
    from vserve.fan import run_daemon
    assert callable(run_daemon)


def test_fan_main_block_fixed_flag():
    """Fan module __main__ block accepts --fixed flag."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "vserve.fan", "--help"],
        capture_output=True, text=True, timeout=5,
    )
    assert "--fixed" in r.stdout
