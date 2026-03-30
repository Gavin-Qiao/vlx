from vx.fan import (
    compute_fan_speed,
    read_state,
    EMERGENCY_TEMP,
    MIN_FAN,
    MAX_CONSECUTIVE_FAILURES,
    FAILURE_RESET_THRESHOLD,
)


class TestComputeFanSpeed:
    """Temperature -> fan speed mapping with quiet-hours cap."""

    # --- Below TEMP_LOW: always minimum ---

    def test_cold_gpu_daytime(self):
        assert compute_fan_speed(temp=30, hour=12) == MIN_FAN

    def test_cold_gpu_nighttime(self):
        assert compute_fan_speed(temp=30, hour=22) == MIN_FAN

    # --- Above TEMP_HIGH: hits cap ---

    def test_hot_gpu_daytime_capped(self):
        assert compute_fan_speed(temp=85, hour=12) == 60

    def test_hot_gpu_nighttime_full(self):
        assert compute_fan_speed(temp=85, hour=22) == 100

    # --- At boundary temps ---

    def test_at_temp_low(self):
        assert compute_fan_speed(temp=35, hour=12) == MIN_FAN

    def test_at_temp_high_daytime(self):
        assert compute_fan_speed(temp=80, hour=12) == 60

    def test_at_temp_high_nighttime(self):
        assert compute_fan_speed(temp=80, hour=22) == 100

    # --- Mid-range: linear interpolation ---

    def test_midpoint_daytime(self):
        speed = compute_fan_speed(temp=57, hour=12)
        assert 43 <= speed <= 47

    def test_midpoint_nighttime(self):
        speed = compute_fan_speed(temp=57, hour=22)
        assert 63 <= speed <= 67

    # --- Quiet-hours boundaries ---

    def test_hour_9_is_quiet(self):
        assert compute_fan_speed(temp=85, hour=9) == 60

    def test_hour_17_is_quiet(self):
        assert compute_fan_speed(temp=85, hour=17) == 60

    def test_hour_18_is_loud(self):
        assert compute_fan_speed(temp=85, hour=18) == 100

    def test_hour_8_is_loud(self):
        assert compute_fan_speed(temp=85, hour=8) == 100

    # --- Custom quiet hours ---

    def test_custom_quiet_hours(self):
        assert compute_fan_speed(temp=85, hour=23, quiet_start=22, quiet_end=6, quiet_max=50) == 50

    def test_custom_quiet_max(self):
        assert compute_fan_speed(temp=85, hour=12, quiet_max=70) == 70

    # --- Emergency override ---

    def test_emergency_override_daytime(self):
        """At emergency temp, ignore quiet cap and go 100%."""
        assert compute_fan_speed(temp=EMERGENCY_TEMP, hour=12) == 100

    def test_emergency_override_nighttime(self):
        assert compute_fan_speed(temp=EMERGENCY_TEMP, hour=22) == 100

    def test_just_below_emergency_daytime_capped(self):
        """Just below emergency temp, quiet cap still applies."""
        assert compute_fan_speed(temp=EMERGENCY_TEMP - 1, hour=12) <= 60

    def test_emergency_with_custom_quiet(self):
        """Emergency overrides even custom quiet settings."""
        assert compute_fan_speed(temp=90, hour=12, quiet_max=40) == 100

    # --- Never below MIN_FAN ---

    def test_never_below_minimum(self):
        for temp in range(0, 100):
            for hour in range(24):
                speed = compute_fan_speed(temp=temp, hour=hour)
                assert speed >= MIN_FAN
                assert speed <= 100


class TestReadState:
    def test_missing_file(self, tmp_path, mocker):
        mocker.patch("vx.fan.STATE_PATH", tmp_path / "nonexistent.json")
        assert read_state() is None

    def test_valid_json(self, tmp_path, mocker):
        f = tmp_path / "state.json"
        f.write_text('{"quiet_start": 22, "quiet_end": 6, "quiet_max": 50}')
        mocker.patch("vx.fan.STATE_PATH", f)
        state = read_state()
        assert state == {"quiet_start": 22, "quiet_end": 6, "quiet_max": 50}

    def test_corrupted_json(self, tmp_path, mocker):
        f = tmp_path / "state.json"
        f.write_text("not json{{{")
        mocker.patch("vx.fan.STATE_PATH", f)
        assert read_state() is None


class TestConstants:
    def test_failure_threshold_ordering(self):
        assert FAILURE_RESET_THRESHOLD < MAX_CONSECUTIVE_FAILURES

    def test_max_failures_is_reasonable(self):
        # 30 failures × 5s interval = 2.5 min before exit
        assert 20 <= MAX_CONSECUTIVE_FAILURES <= 60
