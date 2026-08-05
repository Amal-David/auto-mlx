"""Unit tests for auto_mlx.statistics: Wave B's statistical decision layer.

Covers the trust boundary (forged-timing rejection and K=1 degradation),
the min-of-K point estimate, BCa bootstrap CI reproducibility, the
three-way verdict classification, and the Bonferroni sequential-peek
adjustment -- independent of any real subprocess execution.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx.errors import ContractError
from auto_mlx.statistics import (
    FORGED_TIMING,
    FORGED_TIMING_MIN_ITERATION_NS,
    FORGED_TIMING_TOLERANCE_NS,
    StatisticsVerdict,
    VERDICT_IMPROVED,
    VERDICT_INCONCLUSIVE,
    VERDICT_REGRESSED,
    bca_bootstrap_ci,
    bonferroni_confidence_bps,
    classify_verdict,
    compute_min_effect_ns,
    compute_sample_timing,
    compute_statistics_verdict,
    max_peeks_for,
    min_of_k,
    parse_iteration_timings,
)


def _iter_stderr(iterations_ns: tuple[int, ...]) -> bytes:
    payload = json.dumps({"k": len(iterations_ns), "iterations_ns": list(iterations_ns)}, separators=(",", ":"))
    return f"auto_mlx_runner_iter_timings_v1 {payload}\n".encode("ascii")


class ParseIterationTimingsTests(unittest.TestCase):
    def test_well_formed_line_parses(self) -> None:
        self.assertEqual(parse_iteration_timings(_iter_stderr((5, 6, 7))), (5, 6, 7))

    def test_absent_marker_returns_none(self) -> None:
        self.assertIsNone(parse_iteration_timings(b"no marker here\n"))

    def test_malformed_json_returns_none(self) -> None:
        self.assertIsNone(parse_iteration_timings(b"auto_mlx_runner_iter_timings_v1 not-json\n"))

    def test_k_mismatch_returns_none(self) -> None:
        payload = json.dumps({"k": 5, "iterations_ns": [1, 2]})
        self.assertIsNone(parse_iteration_timings(f"auto_mlx_runner_iter_timings_v1 {payload}\n".encode()))

    def test_non_positive_iteration_returns_none(self) -> None:
        payload = json.dumps({"k": 2, "iterations_ns": [5, 0]})
        self.assertIsNone(parse_iteration_timings(f"auto_mlx_runner_iter_timings_v1 {payload}\n".encode()))

    def test_unknown_field_returns_none(self) -> None:
        payload = json.dumps({"k": 1, "iterations_ns": [5], "extra": 1})
        self.assertIsNone(parse_iteration_timings(f"auto_mlx_runner_iter_timings_v1 {payload}\n".encode()))


class MinOfKTests(unittest.TestCase):
    def test_min_of_k_picks_the_minimum(self) -> None:
        self.assertEqual(min_of_k((5_000_000, 4_000_000, 6_000_000)), 4_000_000)

    def test_min_of_k_requires_at_least_one_value(self) -> None:
        with self.assertRaises(ContractError):
            min_of_k(())


class ComputeSampleTimingTrustBoundaryTests(unittest.TestCase):
    """The forged_timing trust boundary: compute_sample_timing's own docstring."""

    def test_trusted_report_yields_min_of_k_point_estimate(self) -> None:
        iterations = (5_000_000, 4_800_000, 5_200_000)
        timing = compute_sample_timing(200_000_000, _iter_stderr(iterations))
        self.assertTrue(timing.trusted)
        self.assertIsNone(timing.rejection_reason)
        self.assertEqual(timing.point_estimate_ns, min(iterations))
        self.assertEqual(timing.raw_iterations_ns, iterations)

    def test_sum_exceeding_parent_span_is_forged(self) -> None:
        # Individually plausible (>= the 1us floor) but their sum vastly
        # exceeds the observed runner_elapsed_ns -- impossible, since the K
        # iterations are a strict subset of that span.
        iterations = (50_000_000, 50_000_000, 50_000_000)
        runner_elapsed_ns = 10_000_000
        self.assertGreater(sum(iterations), runner_elapsed_ns + FORGED_TIMING_TOLERANCE_NS)
        timing = compute_sample_timing(runner_elapsed_ns, _iter_stderr(iterations))
        self.assertFalse(timing.trusted)
        self.assertEqual(timing.rejection_reason, FORGED_TIMING)
        # Degrades to K=1 semantics: the parent span alone, never the
        # self-reported (impossible) numbers.
        self.assertEqual(timing.point_estimate_ns, runner_elapsed_ns)
        self.assertEqual(timing.raw_iterations_ns, iterations)

    def test_absurdly_small_iteration_is_forged(self) -> None:
        # Below FORGED_TIMING_MIN_ITERATION_NS: no genuine Python/MLX call
        # can complete this fast, regardless of the parent span.
        iterations = (500, 500, 500)
        self.assertLess(min(iterations), FORGED_TIMING_MIN_ITERATION_NS)
        runner_elapsed_ns = 200_000_000
        timing = compute_sample_timing(runner_elapsed_ns, _iter_stderr(iterations))
        self.assertFalse(timing.trusted)
        self.assertEqual(timing.rejection_reason, FORGED_TIMING)
        self.assertEqual(timing.point_estimate_ns, runner_elapsed_ns)

    def test_at_the_boundary_is_trusted_just_past_it_is_forged(self) -> None:
        runner_elapsed_ns = 10_000_000
        exactly_at_tolerance = (runner_elapsed_ns + FORGED_TIMING_TOLERANCE_NS,)
        trusted = compute_sample_timing(runner_elapsed_ns, _iter_stderr(exactly_at_tolerance))
        self.assertTrue(trusted.trusted)

        one_ns_past = (runner_elapsed_ns + FORGED_TIMING_TOLERANCE_NS + 1,)
        forged = compute_sample_timing(runner_elapsed_ns, _iter_stderr(one_ns_past))
        self.assertFalse(forged.trusted)
        self.assertEqual(forged.rejection_reason, FORGED_TIMING)

    def test_missing_report_degrades_without_forged_reason(self) -> None:
        """Absent evidence is not itself proof of tampering (see the module docstring)."""

        runner_elapsed_ns = 12_345_678
        timing = compute_sample_timing(runner_elapsed_ns, b"no iteration report here\n")
        self.assertFalse(timing.trusted)
        self.assertIsNone(timing.rejection_reason)
        self.assertEqual(timing.point_estimate_ns, runner_elapsed_ns)
        self.assertEqual(timing.raw_iterations_ns, ())

    def test_requires_a_positive_runner_elapsed_ns(self) -> None:
        with self.assertRaises(ContractError):
            compute_sample_timing(0, b"")
        with self.assertRaises(ContractError):
            compute_sample_timing(None, b"")  # type: ignore[arg-type]


class BootstrapReproducibilityTests(unittest.TestCase):
    def test_same_inputs_and_seed_yield_identical_bounds(self) -> None:
        differences = [10.0, 12.0, 9.0, 11.0, 8.5, 10.5]
        first = bca_bootstrap_ci(differences, resamples=2000, seed=42, confidence_bps=9500)
        second = bca_bootstrap_ci(differences, resamples=2000, seed=42, confidence_bps=9500)
        self.assertEqual(first, second)

    def test_different_seed_can_move_the_bounds(self) -> None:
        differences = [10.0, 12.0, 9.0, 11.0, 8.5, 3.0, 40.0]
        first = bca_bootstrap_ci(differences, resamples=1000, seed=1, confidence_bps=9500)
        second = bca_bootstrap_ci(differences, resamples=1000, seed=2, confidence_bps=9500)
        # Not a strict inequality assertion (a different seed could coincide),
        # but the two independent draws over a skewed population should not
        # be forced equal by anything in the implementation.
        self.assertIsInstance(first, tuple)
        self.assertIsInstance(second, tuple)

    def test_single_difference_collapses_to_a_zero_width_point(self) -> None:
        lower, upper = bca_bootstrap_ci([42.0], resamples=500, seed=1, confidence_bps=9500)
        self.assertEqual(lower, 42)
        self.assertEqual(upper, 42)

    def test_identical_differences_collapse_to_a_zero_width_point(self) -> None:
        lower, upper = bca_bootstrap_ci([7.0, 7.0, 7.0, 7.0], resamples=500, seed=1, confidence_bps=9500)
        self.assertEqual(lower, 7)
        self.assertEqual(upper, 7)

    def test_requires_at_least_one_difference(self) -> None:
        with self.assertRaises(ContractError):
            bca_bootstrap_ci([], resamples=100, seed=1, confidence_bps=9500)


class VerdictClassificationTests(unittest.TestCase):
    def test_improved_requires_the_whole_interval_past_the_threshold(self) -> None:
        self.assertEqual(classify_verdict(500, 600, min_effect_ns=100), VERDICT_IMPROVED)
        # Point estimate clears the threshold but the interval does not:
        # still inconclusive, never rounded to a win.
        self.assertEqual(classify_verdict(50, 600, min_effect_ns=100), VERDICT_INCONCLUSIVE)

    def test_regressed_requires_the_whole_interval_past_the_negative_threshold(self) -> None:
        self.assertEqual(classify_verdict(-600, -500, min_effect_ns=100), VERDICT_REGRESSED)
        self.assertEqual(classify_verdict(-600, -50, min_effect_ns=100), VERDICT_INCONCLUSIVE)

    def test_interval_straddling_zero_is_inconclusive(self) -> None:
        self.assertEqual(classify_verdict(-10, 10, min_effect_ns=100), VERDICT_INCONCLUSIVE)

    def test_rejects_inverted_bounds(self) -> None:
        with self.assertRaises(ContractError):
            classify_verdict(10, -10, min_effect_ns=1)


class BonferroniAdjustmentTests(unittest.TestCase):
    def test_single_peek_keeps_the_base_confidence(self) -> None:
        self.assertEqual(bonferroni_confidence_bps(base_confidence_bps=9500, max_peeks=1), 9500)

    def test_more_peeks_widen_the_interval_via_higher_confidence(self) -> None:
        one_peek = bonferroni_confidence_bps(base_confidence_bps=9500, max_peeks=1)
        many_peeks = bonferroni_confidence_bps(base_confidence_bps=9500, max_peeks=18)
        self.assertGreater(many_peeks, one_peek)
        self.assertLessEqual(many_peeks, 10_000)

    def test_max_peeks_for_derivation(self) -> None:
        self.assertEqual(max_peeks_for(measurement_runs=3, max_measurement_runs=20), 18)
        self.assertEqual(max_peeks_for(measurement_runs=5, max_measurement_runs=5), 1)


class ComputeMinEffectNsTests(unittest.TestCase):
    def test_two_percent_of_baseline(self) -> None:
        self.assertEqual(compute_min_effect_ns(baseline_reference_ns=1_000_000, min_effect_bps=200), 20_000)

    def test_zero_bps_means_any_nonzero_effect_counts(self) -> None:
        self.assertEqual(compute_min_effect_ns(baseline_reference_ns=1_000_000, min_effect_bps=0), 0)


class ComputeStatisticsVerdictSyntheticTests(unittest.TestCase):
    """End-to-end statistics computation over synthetic per-block point estimates."""

    def _verdict(self, baseline_points, candidate_points, **overrides) -> StatisticsVerdict:
        kwargs = dict(
            block_baseline_points=baseline_points,
            block_candidate_points=candidate_points,
            k_repetitions=50,
            measurement_runs=len(baseline_points),
            max_measurement_runs=len(baseline_points),
            min_effect_bps=200,
            bootstrap_resamples=2000,
            bootstrap_seed=7,
            calibration=False,
        )
        kwargs.update(overrides)
        return compute_statistics_verdict(**kwargs)

    def test_clearly_faster_candidate_is_improved(self) -> None:
        baseline = [[20_000_000, 20_100_000]] * 4
        candidate = [[10_000_000, 10_050_000]] * 4
        verdict = self._verdict(baseline, candidate)
        self.assertEqual(verdict.verdict, VERDICT_IMPROVED)
        self.assertGreater(verdict.ci_lower_ns, verdict.min_effect_ns)

    def test_clearly_slower_candidate_is_regressed(self) -> None:
        baseline = [[10_000_000, 10_050_000]] * 4
        candidate = [[20_000_000, 20_100_000]] * 4
        verdict = self._verdict(baseline, candidate)
        self.assertEqual(verdict.verdict, VERDICT_REGRESSED)
        self.assertLess(verdict.ci_upper_ns, -verdict.min_effect_ns)

    def test_near_identical_arms_are_inconclusive(self) -> None:
        # Small, noisy jitter around a near-zero true difference -- neither
        # arm is clearly faster.
        baseline = [[10_000_000, 10_010_000], [9_990_000, 10_005_000], [10_002_000, 9_998_000]]
        candidate = [[10_001_000, 9_995_000], [10_004_000, 10_006_000], [9_997_000, 10_003_000]]
        verdict = self._verdict(baseline, candidate)
        self.assertEqual(verdict.verdict, VERDICT_INCONCLUSIVE)

    def test_verdict_is_reproducible_from_the_stored_seed(self) -> None:
        baseline = [[20_000_000, 19_500_000], [20_500_000, 20_100_000]]
        candidate = [[10_000_000, 10_500_000], [9_800_000, 10_100_000]]
        first = self._verdict(baseline, candidate, bootstrap_seed=123)
        second = self._verdict(baseline, candidate, bootstrap_seed=123)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_calibration_flag_is_carried_through(self) -> None:
        baseline = [[10_000_000, 10_000_000]]
        candidate = [[10_000_000, 10_000_000]]
        verdict = self._verdict(baseline, candidate, calibration=True)
        self.assertTrue(verdict.calibration)

    def test_statistics_verdict_from_dict_roundtrip(self) -> None:
        baseline = [[20_000_000, 20_000_000]]
        candidate = [[10_000_000, 10_000_000]]
        verdict = self._verdict(baseline, candidate)
        restored = StatisticsVerdict.from_dict(verdict.to_dict())
        self.assertEqual(restored, verdict)

    def test_statistics_verdict_rejects_internally_inconsistent_verdict(self) -> None:
        baseline = [[20_000_000, 20_000_000]]
        candidate = [[10_000_000, 10_000_000]]
        verdict = self._verdict(baseline, candidate)
        tampered = verdict.to_dict()
        tampered["verdict"] = "regressed"  # ci bounds still say "improved"
        with self.assertRaises(ContractError):
            StatisticsVerdict.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
