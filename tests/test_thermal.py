from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx.thermal import ThermalReading, parse_pmset_therm, probe_thermal_pressure, thermal_preflight


class PmsetParseRobustnessTests(unittest.TestCase):
    """Canned-output coverage: this parser must be pure and total, never raise."""

    def test_no_recorded_thermal_data_is_unknown(self) -> None:
        # The exact shape observed on a real, currently-nominal Apple
        # Silicon Mac: pmset has nothing to report yet.
        output = (
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n"
            "Note: No CPU power status has been recorded\n"
        )
        reading = parse_pmset_therm(output)
        self.assertEqual(reading.state, "unknown")
        self.assertIsNone(reading.cpu_speed_limit_percent)
        self.assertIsNone(reading.cpu_scheduler_limit_percent)
        self.assertIsNone(reading.thermal_pressure_level)

    def test_empty_output_is_unknown(self) -> None:
        self.assertEqual(parse_pmset_therm("").state, "unknown")

    def test_garbage_output_is_unknown_not_a_crash(self) -> None:
        reading = parse_pmset_therm("\x00\x01 not pmset output at all {}][;'\n\n\tsome garbage\x99")
        self.assertEqual(reading.state, "unknown")

    def test_non_string_input_is_coerced_not_raised(self) -> None:
        reading = parse_pmset_therm(12345)  # type: ignore[arg-type]
        self.assertEqual(reading.state, "unknown")

    def test_cpu_speed_limit_100_is_nominal(self) -> None:
        reading = parse_pmset_therm("CPU_Speed_Limit         =       100\n")
        self.assertEqual(reading.state, "nominal")
        self.assertEqual(reading.cpu_speed_limit_percent, 100)

    def test_cpu_speed_limit_below_100_is_throttled(self) -> None:
        reading = parse_pmset_therm("CPU_Speed_Limit         =       45\n")
        self.assertEqual(reading.state, "throttled")
        self.assertEqual(reading.cpu_speed_limit_percent, 45)

    def test_cpu_scheduler_limit_below_100_is_throttled(self) -> None:
        reading = parse_pmset_therm("CPU_Scheduler_Limit = 60\n")
        self.assertEqual(reading.state, "throttled")
        self.assertEqual(reading.cpu_scheduler_limit_percent, 60)

    def test_thermal_pressure_level_nominal_is_nominal(self) -> None:
        reading = parse_pmset_therm("Thermal_Pressure_Level = Nominal\n")
        self.assertEqual(reading.state, "nominal")
        self.assertEqual(reading.thermal_pressure_level, "Nominal")

    def test_thermal_pressure_level_heavy_is_throttled(self) -> None:
        reading = parse_pmset_therm("Thermal_Pressure_Level = Heavy\n")
        self.assertEqual(reading.state, "throttled")

    def test_mixed_fields_any_degraded_field_is_throttled(self) -> None:
        output = "CPU_Speed_Limit = 100\nCPU_Scheduler_Limit = 40\n"
        reading = parse_pmset_therm(output)
        self.assertEqual(reading.state, "throttled")
        self.assertEqual(reading.cpu_speed_limit_percent, 100)
        self.assertEqual(reading.cpu_scheduler_limit_percent, 40)

    def test_detail_is_bounded_length(self) -> None:
        reading = parse_pmset_therm("CPU_Speed_Limit = 100\n" + ("x" * 10_000))
        self.assertLessEqual(len(reading.detail), 2000)

    def test_reading_round_trips_through_dict(self) -> None:
        reading = parse_pmset_therm("CPU_Speed_Limit = 45\n")
        self.assertEqual(ThermalReading.from_dict(reading.to_dict()), reading)

    def test_invalid_state_is_rejected_by_the_dataclass_itself(self) -> None:
        with self.assertRaises(ValueError):
            ThermalReading("not-a-real-state", None, None, None, "")


class ProbeThermalPressureTests(unittest.TestCase):
    def test_missing_binary_degrades_to_unknown_not_a_crash(self) -> None:
        reading = probe_thermal_pressure(pmset_path="/definitely/not/a/real/pmset/binary")
        self.assertEqual(reading.state, "unknown")

    def test_timeout_degrades_to_unknown_not_a_crash(self) -> None:
        # A real command that never exits within an impossibly tight budget.
        reading = probe_thermal_pressure(pmset_path=sys.executable, timeout_seconds=0.0001)
        self.assertIn(reading.state, {"unknown"})

    def test_real_pmset_probe_never_raises(self) -> None:
        # Exercises the real host binary if present; on non-macOS CI this
        # degrades to "unknown" via FileNotFoundError, never a crash.
        reading = probe_thermal_pressure()
        self.assertIn(reading.state, {"nominal", "throttled", "unknown"})


class ThermalPreflightTests(unittest.TestCase):
    def test_nominal_reading_never_retries(self) -> None:
        calls = []

        def prober() -> ThermalReading:
            calls.append(1)
            return ThermalReading("nominal", 100, None, None, "")

        sleeps = []
        result = thermal_preflight(prober=prober, sleep=sleeps.append, retry_pause_seconds=30.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sleeps, [])
        self.assertFalse(result["retried"])
        self.assertFalse(result["thermally_suspect"])

    def test_unknown_reading_never_retries_and_is_not_suspect(self) -> None:
        calls = []

        def prober() -> ThermalReading:
            calls.append(1)
            return ThermalReading("unknown", None, None, None, "")

        sleeps = []
        result = thermal_preflight(prober=prober, sleep=sleeps.append)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sleeps, [])
        self.assertFalse(result["thermally_suspect"])

    def test_throttled_then_recovered_retries_once_and_is_not_suspect(self) -> None:
        readings = iter([ThermalReading("throttled", 50, None, None, ""), ThermalReading("nominal", 100, None, None, "")])

        def prober() -> ThermalReading:
            return next(readings)

        sleeps = []
        result = thermal_preflight(prober=prober, sleep=sleeps.append, retry_pause_seconds=30.0)
        self.assertEqual(sleeps, [30.0])
        self.assertTrue(result["retried"])
        self.assertFalse(result["thermally_suspect"])
        self.assertEqual(result["initial"]["state"], "throttled")
        self.assertEqual(result["final"]["state"], "nominal")

    def test_still_throttled_after_retry_is_suspect(self) -> None:
        def prober() -> ThermalReading:
            return ThermalReading("throttled", 50, None, None, "")

        result = thermal_preflight(prober=prober, sleep=lambda seconds: None)
        self.assertTrue(result["retried"])
        self.assertTrue(result["thermally_suspect"])

    def test_result_is_json_safe(self) -> None:
        import json

        result = thermal_preflight(prober=lambda: ThermalReading("nominal", 100, None, None, ""), sleep=lambda s: None)
        json.dumps(result)  # must not raise


if __name__ == "__main__":
    unittest.main()
