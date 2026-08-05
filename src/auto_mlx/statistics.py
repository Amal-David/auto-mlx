"""Wave B statistically-sound accept/reject decisions.

Apple Silicon offers no frequency lock or core pinning a local evaluator can
reach without elevated privileges (see ``auto_mlx.thermal``).  This module
is where "statistics carry the load" instead: it turns per-sample,
self-reported in-runner iteration timings into a trustworthy per-sample
point estimate, and turns a set of paired per-block differences into a
BCa-bootstrap confidence interval and a closed three-way verdict
(``improved`` / ``regressed`` / ``inconclusive``).

Trust boundary
--------------

A runner subprocess (see ``auto_mlx.runners.reference_matmul``) may report
its own per-iteration timings on a stderr channel, but that channel is
self-reported and is never evidentiary on its own.  The parent-observed
``ExecutionRecord.runner_elapsed_ns`` -- the evaluator's own wall-clock
observation of the runner subprocess's launch-to-exit span -- remains the
sole evidentiary timing anchor.  :func:`compute_sample_timing` cross-checks
a reported array against that anchor before trusting it at all; an
implausible report (impossible or absurdly small relative to the observed
span) is a typed ``forged_timing`` rejection that degrades the sample to
K=1 semantics (the parent span alone) rather than being silently trusted.

Everything here is pure stdlib (``random``, ``statistics.NormalDist``,
``json``) -- no third-party statistics dependency.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Final

from .errors import ContractError, FailureCode


STATISTICS_SCHEMA: Final = "auto_mlx.statistics.v1"

# Must stay byte-identical to auto_mlx.runners.reference_matmul.ITER_TIMINGS_MARKER
# (that module cannot import auto_mlx -- see its own docstring).
_ITER_TIMINGS_MARKER: Final = "auto_mlx_runner_iter_timings_v1"
_ITER_TIMINGS_PREFIX: Final = (_ITER_TIMINGS_MARKER + " ").encode("ascii")

# Forged-timing tolerance, in nanoseconds.  See compute_sample_timing's
# docstring for the reasoning behind both constants.
FORGED_TIMING_TOLERANCE_NS: Final = 2_000_000
FORGED_TIMING_MIN_ITERATION_NS: Final = 1_000

# The base (pre-sequential-adjustment) two-sided confidence level, in basis
# points out of 10,000.  95%.  Bonferroni-adjusted per peek by
# bonferroni_confidence_bps below.
BASE_CONFIDENCE_BPS: Final = 9_500

VERDICT_IMPROVED: Final = "improved"
VERDICT_REGRESSED: Final = "regressed"
VERDICT_INCONCLUSIVE: Final = "inconclusive"
VERDICTS: Final = frozenset({VERDICT_IMPROVED, VERDICT_REGRESSED, VERDICT_INCONCLUSIVE})

FORGED_TIMING: Final = "forged_timing"

_STATISTICS_VERDICT_FIELDS: Final = frozenset(
    {
        "schema", "k_repetitions", "block_count_used", "verdict", "ci_lower_ns", "ci_upper_ns",
        "min_effect_ns", "baseline_reference_ns", "bootstrap_seed", "bootstrap_resamples",
        "confidence_bps", "calibration",
    }
)


def parse_iteration_timings(stderr: bytes) -> tuple[int, ...] | None:
    """Extract and strictly validate the runner's self-reported iteration array.

    Returns ``None`` -- never raises -- for anything that is not exactly a
    well-formed ``{"k": K, "iterations_ns": [K positive integers]}`` line
    behind :data:`_ITER_TIMINGS_MARKER`: absent marker, malformed JSON,
    wrong shape, non-positive or non-integer entries, or a ``k`` that does
    not match the array length.  A missing/malformed report is not itself
    evidence of tampering -- it simply means no self-reported enhancement is
    available, and callers degrade to K=1 (parent-span) semantics.
    """

    if type(stderr) is not bytes:
        return None
    for line in stderr.split(b"\n"):
        if not line.startswith(_ITER_TIMINGS_PREFIX):
            continue
        payload = line[len(_ITER_TIMINGS_PREFIX):]
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if type(parsed) is not dict or set(parsed) != {"k", "iterations_ns"}:
            return None
        k = parsed["k"]
        iterations = parsed["iterations_ns"]
        if type(k) is not int or k < 1:
            return None
        if type(iterations) is not list or len(iterations) != k:
            return None
        if any(type(value) is not int or value <= 0 for value in iterations):
            return None
        return tuple(iterations)
    return None


@dataclass(frozen=True, slots=True)
class SampleTiming:
    """One sample's timing evidence and the point estimate derived from it."""

    raw_iterations_ns: tuple[int, ...]
    trusted: bool
    point_estimate_ns: int
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if type(self.raw_iterations_ns) is not tuple or any(
            type(value) is not int or value <= 0 for value in self.raw_iterations_ns
        ):
            raise ContractError("raw_iterations_ns must be a tuple of positive integers", code=FailureCode.WRONG_TYPE)
        if type(self.trusted) is not bool:
            raise ContractError("trusted must be a boolean", code=FailureCode.WRONG_TYPE)
        if type(self.point_estimate_ns) is not int or self.point_estimate_ns <= 0:
            raise ContractError("point_estimate_ns must be a positive integer", code=FailureCode.WRONG_TYPE)
        if self.rejection_reason is not None and self.rejection_reason != FORGED_TIMING:
            raise ContractError("rejection_reason must be null or forged_timing", code=FailureCode.INVALID_VALUE)
        if self.trusted and self.rejection_reason is not None:
            raise ContractError("a trusted sample cannot carry a rejection_reason", code=FailureCode.INVALID_VALUE)
        if self.rejection_reason is not None and not self.raw_iterations_ns:
            raise ContractError("forged_timing requires the reported iterations that triggered it", code=FailureCode.INVALID_VALUE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_iterations_ns": list(self.raw_iterations_ns),
            "trusted": self.trusted,
            "point_estimate_ns": self.point_estimate_ns,
            "rejection_reason": self.rejection_reason,
        }


def min_of_k(iterations_ns: Sequence[int]) -> int:
    """Per-sample point estimate from K repeated timings: the minimum.

    Timing noise on a shared, unpinned machine is one-sided -- scheduling
    preemption, thermal throttling, and page faults can only make an
    iteration slower than its true cost, never faster (the Triton/pyperf
    convention; TVM's mean-of-repeats is the documented counter-evidence,
    noted here as a design choice, not an oversight).  The minimum is the
    closest a bounded sample gets to an unperturbed measurement.
    """

    if not iterations_ns:
        raise ContractError("min_of_k requires at least one iteration", code=FailureCode.INVALID_VALUE)
    return min(iterations_ns)


def compute_sample_timing(runner_elapsed_ns: int | None, stderr: bytes) -> SampleTiming:
    """Pure function of one sample's evidence: cross-check, then reduce.

    Trust rule (non-negotiable): ``runner_elapsed_ns`` -- the parent's own
    observation of the runner subprocess's launch-to-exit span -- is the
    sole evidentiary anchor.  A self-reported iteration array is accepted
    only when it survives two checks against that anchor:

    1. **Impossible / too large.**  ``sum(iterations_ns)`` cannot exceed
       ``runner_elapsed_ns`` -- the K measured iterations are a strict
       subset of the runner subprocess's own observed lifetime (which also
       includes interpreter startup, the ``mlx`` import, the uncounted
       warmup pass, and stdout/stderr I/O).  ``FORGED_TIMING_TOLERANCE_NS``
       (2ms) is a deliberately small allowance for the two clocks involved
       (the parent's ``time.monotonic_ns()`` and the child's
       ``time.perf_counter_ns()``) not being read at the exact same
       instant, not a claim that they can meaningfully disagree.
    2. **Absurdly small.**  No genuine Python-level MLX call, ``mx.eval``,
       and function-call overhead can complete in under
       ``FORGED_TIMING_MIN_ITERATION_NS`` (1 microsecond) on any real
       hardware.  A reported iteration below that floor cannot be a
       genuine measurement of real computation, regardless of workload --
       this check is intentionally workload-agnostic (it does not assume
       anything about the toy-matmul's expected cost).

    A report that fails either check -- or is absent/malformed (see
    :func:`parse_iteration_timings`) -- degrades the sample to K=1
    semantics: the point estimate becomes ``runner_elapsed_ns`` itself
    (already fully evidentiary on its own, per Wave A), never a
    self-reported number.  Only a report that survives both checks yields a
    min-of-K point estimate (:func:`min_of_k`).
    """

    if type(runner_elapsed_ns) is not int or runner_elapsed_ns <= 0:
        raise ContractError("compute_sample_timing requires a positive runner_elapsed_ns", code=FailureCode.INVALID_VALUE)
    reported = parse_iteration_timings(stderr)
    if reported is None:
        return SampleTiming((), False, runner_elapsed_ns, None)
    total = sum(reported)
    smallest = min(reported)
    implausible = total > runner_elapsed_ns + FORGED_TIMING_TOLERANCE_NS or smallest < FORGED_TIMING_MIN_ITERATION_NS
    if implausible:
        return SampleTiming(reported, False, runner_elapsed_ns, FORGED_TIMING)
    return SampleTiming(reported, True, min_of_k(reported), None)


def bonferroni_confidence_bps(*, base_confidence_bps: int, max_peeks: int) -> int:
    """Bonferroni-style sequential-peek adjustment, as an integer bps level.

    This codebase forbids floating-point values on any wire (see
    ``auto_mlx.canonical``), so the confidence level actually used is
    computed once, as an integer basis-points value, and that integer -- not
    an unrounded float -- is what both the live evaluation and any later
    independent recomputation use to derive the bootstrap alpha.  This
    fixes a single confidence level for every peek across the whole
    sequential procedure (``max_peeks`` is the policy-declared worst case:
    ``max_measurement_runs - measurement_runs + 1``, fixed before the first
    block is measured), rather than tightening it further as peeks are
    actually consumed.  That is deliberately conservative: it guarantees a
    Bonferroni family-wise error bound of ``base_alpha`` across up to
    ``max_peeks`` sequential looks, whichever peek turns out decisive, with
    a closed-form, trivially reproducible formula (unlike an O'Brien-Fleming
    /Pocock boundary table, which has no simple closed form).
    """

    if type(base_confidence_bps) is not int or not (0 <= base_confidence_bps <= 10_000):
        raise ContractError("base_confidence_bps must be an integer in [0, 10000]", code=FailureCode.INVALID_VALUE)
    if type(max_peeks) is not int or max_peeks < 1:
        raise ContractError("max_peeks must be a positive integer", code=FailureCode.INVALID_VALUE)
    base_alpha_bps = 10_000 - base_confidence_bps
    adjusted_alpha_bps = round(base_alpha_bps / max_peeks)
    return max(0, min(10_000, 10_000 - adjusted_alpha_bps))


def bca_bootstrap_ci(
    differences: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence_bps: int,
) -> tuple[int, int]:
    """BCa (bias-corrected and accelerated) bootstrap CI, rounded to ns.

    Pure stdlib: resampling uses ``random.Random(seed)`` (a fresh instance
    per call, so the result is a pure function of ``(differences, resamples,
    seed, confidence_bps)`` and independent of any earlier peek's RNG
    state); the normal quantile/CDF functions use
    ``statistics.NormalDist``. Deterministic given the same inputs on the
    same Python build -- the reproducibility contract Wave B's supervisor
    recomputation relies on (see ``auto_mlx.receipts``).

    Bounds are rounded to the nearest integer nanosecond (Python's
    round-half-to-even) because this codebase forbids floats on any wire;
    recomputation redoes the identical float arithmetic and rounds the
    identical way.
    """

    n = len(differences)
    if n < 1:
        raise ContractError("bootstrap requires at least one paired difference", code=FailureCode.INVALID_VALUE)
    if type(resamples) is not int or resamples < 1:
        raise ContractError("resamples must be a positive integer", code=FailureCode.INVALID_VALUE)
    if type(seed) is not int or seed < 0:
        raise ContractError("seed must be a non-negative integer", code=FailureCode.INVALID_VALUE)
    if type(confidence_bps) is not int or not (0 <= confidence_bps <= 10_000):
        raise ContractError("confidence_bps must be an integer in [0, 10000]", code=FailureCode.INVALID_VALUE)

    values = [float(value) for value in differences]
    theta_hat = sum(values) / n

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()

    normal = NormalDist()
    # Bias correction z0: how far the observed statistic sits inside its
    # own bootstrap distribution.  Ties count as half-below (standard BCa
    # treatment); clamp away from the open interval's ends so inv_cdf never
    # sees exactly 0 or 1.
    less = sum(1 for value in boot_means if value < theta_hat)
    tied = sum(1 for value in boot_means if value == theta_hat)
    proportion = (less + tied / 2.0) / resamples
    epsilon = 1.0 / (resamples * 4)
    proportion = min(max(proportion, epsilon), 1.0 - epsilon)
    z0 = normal.inv_cdf(proportion)

    # Acceleration a: jackknife-estimated skewness of the statistic.  With a
    # single difference (n == 1) the jackknife has no leave-one-out
    # variance, so acceleration is undefined and treated as zero (no
    # skewness correction) rather than raising.
    if n > 1:
        jackknife_means = []
        for index in range(n):
            leave_one_out = values[:index] + values[index + 1 :]
            jackknife_means.append(sum(leave_one_out) / len(leave_one_out))
        jack_mean = sum(jackknife_means) / n
        numerator = sum((jack_mean - value) ** 3 for value in jackknife_means)
        denominator = 6.0 * (sum((jack_mean - value) ** 2 for value in jackknife_means) ** 1.5)
        acceleration = numerator / denominator if denominator != 0.0 else 0.0
    else:
        acceleration = 0.0

    alpha = 1.0 - confidence_bps / 10_000.0
    z_lo = normal.inv_cdf(alpha / 2.0)
    z_hi = normal.inv_cdf(1.0 - alpha / 2.0)

    def _adjusted_percentile(z: float) -> float:
        denom = 1.0 - acceleration * (z0 + z)
        adjusted_z = z0 + (z0 + z) / denom if denom != 0.0 else z0 + z
        return min(max(normal.cdf(adjusted_z), 0.0), 1.0)

    index_lo = min(max(int(_adjusted_percentile(z_lo) * resamples), 0), resamples - 1)
    index_hi = min(max(int(_adjusted_percentile(z_hi) * resamples), 0), resamples - 1)
    if index_hi < index_lo:
        index_lo, index_hi = index_hi, index_lo
    return round(boot_means[index_lo]), round(boot_means[index_hi])


def classify_verdict(ci_lower_ns: int, ci_upper_ns: int, *, min_effect_ns: int) -> str:
    """Three-way verdict from a CI and a minimum-effect threshold.

    ``differences`` are defined as ``baseline_point_estimate -
    candidate_point_estimate``, so a positive difference means the
    candidate is faster.  ``improved`` requires the WHOLE interval to clear
    the threshold (not just the point estimate); ``inconclusive`` is a
    first-class outcome, never rounded to a win or a loss.
    """

    if type(min_effect_ns) is not int or min_effect_ns < 0:
        raise ContractError("min_effect_ns must be a non-negative integer", code=FailureCode.INVALID_VALUE)
    if ci_lower_ns > ci_upper_ns:
        raise ContractError("ci_lower_ns cannot exceed ci_upper_ns", code=FailureCode.INVALID_VALUE)
    if ci_lower_ns > min_effect_ns:
        return VERDICT_IMPROVED
    if ci_upper_ns < -min_effect_ns:
        return VERDICT_REGRESSED
    return VERDICT_INCONCLUSIVE


@dataclass(frozen=True, slots=True)
class StatisticsVerdict:
    """The complete, self-describing statistical decision for one evaluation."""

    k_repetitions: int
    block_count_used: int
    verdict: str
    ci_lower_ns: int
    ci_upper_ns: int
    min_effect_ns: int
    baseline_reference_ns: int
    bootstrap_seed: int
    bootstrap_resamples: int
    confidence_bps: int
    calibration: bool

    def __post_init__(self) -> None:
        if type(self.k_repetitions) is not int or self.k_repetitions < 1:
            raise ContractError("k_repetitions must be a positive integer", code=FailureCode.INVALID_VALUE)
        if type(self.block_count_used) is not int or self.block_count_used < 1:
            raise ContractError("block_count_used must be a positive integer", code=FailureCode.INVALID_VALUE)
        if self.verdict not in VERDICTS:
            raise ContractError("verdict is not a closed statistics verdict", code=FailureCode.INVALID_VALUE)
        if type(self.ci_lower_ns) is not int or type(self.ci_upper_ns) is not int or self.ci_lower_ns > self.ci_upper_ns:
            raise ContractError("ci bounds must be integers with ci_lower_ns <= ci_upper_ns", code=FailureCode.INVALID_VALUE)
        if type(self.min_effect_ns) is not int or self.min_effect_ns < 0:
            raise ContractError("min_effect_ns must be a non-negative integer", code=FailureCode.INVALID_VALUE)
        if type(self.baseline_reference_ns) is not int or self.baseline_reference_ns <= 0:
            raise ContractError("baseline_reference_ns must be a positive integer", code=FailureCode.INVALID_VALUE)
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ContractError("bootstrap_seed must be a non-negative integer", code=FailureCode.INVALID_VALUE)
        if type(self.bootstrap_resamples) is not int or self.bootstrap_resamples < 1:
            raise ContractError("bootstrap_resamples must be a positive integer", code=FailureCode.INVALID_VALUE)
        if type(self.confidence_bps) is not int or not (0 <= self.confidence_bps <= 10_000):
            raise ContractError("confidence_bps must be an integer in [0, 10000]", code=FailureCode.INVALID_VALUE)
        if type(self.calibration) is not bool:
            raise ContractError("calibration must be a boolean", code=FailureCode.WRONG_TYPE)
        expected_verdict = classify_verdict(self.ci_lower_ns, self.ci_upper_ns, min_effect_ns=self.min_effect_ns)
        if expected_verdict != self.verdict:
            raise ContractError("verdict is not derived from its own ci bounds and threshold", code=FailureCode.IDENTITY_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATISTICS_SCHEMA,
            "k_repetitions": self.k_repetitions,
            "block_count_used": self.block_count_used,
            "verdict": self.verdict,
            "ci_lower_ns": self.ci_lower_ns,
            "ci_upper_ns": self.ci_upper_ns,
            "min_effect_ns": self.min_effect_ns,
            "baseline_reference_ns": self.baseline_reference_ns,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence_bps": self.confidence_bps,
            "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StatisticsVerdict":
        if type(value) is not dict:
            raise ContractError("statistics verdict must be a JSON object", code=FailureCode.WRONG_TYPE)
        if any(type(key) is not str for key in value):
            raise ContractError("statistics verdict field names must be strings", code=FailureCode.WRONG_TYPE)
        unknown = set(value) - _STATISTICS_VERDICT_FIELDS
        missing = _STATISTICS_VERDICT_FIELDS - set(value)
        if unknown:
            raise ContractError(f"statistics verdict has unknown field(s): {sorted(unknown)}", code=FailureCode.INVALID_VALUE)
        if missing:
            raise ContractError(f"statistics verdict is missing field(s): {sorted(missing)}", code=FailureCode.INVALID_VALUE)
        if value["schema"] != STATISTICS_SCHEMA:
            raise ContractError("statistics verdict schema is incompatible", code=FailureCode.INVALID_VALUE)
        return cls(
            k_repetitions=value["k_repetitions"],
            block_count_used=value["block_count_used"],
            verdict=value["verdict"],
            ci_lower_ns=value["ci_lower_ns"],
            ci_upper_ns=value["ci_upper_ns"],
            min_effect_ns=value["min_effect_ns"],
            baseline_reference_ns=value["baseline_reference_ns"],
            bootstrap_seed=value["bootstrap_seed"],
            bootstrap_resamples=value["bootstrap_resamples"],
            confidence_bps=value["confidence_bps"],
            calibration=value["calibration"],
        )


def max_peeks_for(*, measurement_runs: int, max_measurement_runs: int) -> int:
    if max_measurement_runs < measurement_runs:
        raise ContractError("max_measurement_runs must be >= measurement_runs", code=FailureCode.INVALID_POLICY)
    return max_measurement_runs - measurement_runs + 1


def compute_min_effect_ns(*, baseline_reference_ns: int, min_effect_bps: int) -> int:
    if type(baseline_reference_ns) is not int or baseline_reference_ns <= 0:
        raise ContractError("baseline_reference_ns must be a positive integer", code=FailureCode.INVALID_VALUE)
    if type(min_effect_bps) is not int or not (0 <= min_effect_bps <= 10_000):
        raise ContractError("min_effect_bps must be an integer in [0, 10000]", code=FailureCode.INVALID_VALUE)
    return (baseline_reference_ns * min_effect_bps) // 10_000


def compute_statistics_verdict(
    *,
    block_baseline_points: Sequence[Sequence[int]],
    block_candidate_points: Sequence[Sequence[int]],
    k_repetitions: int,
    measurement_runs: int,
    max_measurement_runs: int,
    min_effect_bps: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    calibration: bool,
) -> StatisticsVerdict:
    """Compute the full Wave B verdict from per-block point-estimate pairs.

    ``block_baseline_points``/``block_candidate_points`` are one entry per
    measured block (ABBA/BAAB), each a sequence of that block's own arm's
    sample point estimates (2 per block, per the fixed paired-block
    shape). The paired per-block difference is
    ``mean(baseline_points) - mean(candidate_points)`` (positive means the
    candidate is faster); those differences are the population the BCa
    bootstrap resamples.
    """

    block_count = len(block_baseline_points)
    if block_count < 1 or len(block_candidate_points) != block_count:
        raise ContractError("block point estimates must be non-empty and arm-paired", code=FailureCode.INVALID_VALUE)
    block_diffs: list[float] = []
    baseline_totals: list[int] = []
    for baseline_points, candidate_points in zip(block_baseline_points, block_candidate_points):
        if not baseline_points or not candidate_points:
            raise ContractError("every block needs at least one point estimate per arm", code=FailureCode.INVALID_VALUE)
        baseline_mean = sum(baseline_points) / len(baseline_points)
        candidate_mean = sum(candidate_points) / len(candidate_points)
        block_diffs.append(baseline_mean - candidate_mean)
        baseline_totals.extend(baseline_points)

    baseline_reference_ns = sum(baseline_totals) // len(baseline_totals)
    min_effect_ns = compute_min_effect_ns(baseline_reference_ns=baseline_reference_ns, min_effect_bps=min_effect_bps)
    max_peeks = max_peeks_for(measurement_runs=measurement_runs, max_measurement_runs=max_measurement_runs)
    confidence_bps = bonferroni_confidence_bps(base_confidence_bps=BASE_CONFIDENCE_BPS, max_peeks=max_peeks)
    ci_lower_ns, ci_upper_ns = bca_bootstrap_ci(
        block_diffs, resamples=bootstrap_resamples, seed=bootstrap_seed, confidence_bps=confidence_bps
    )
    verdict = classify_verdict(ci_lower_ns, ci_upper_ns, min_effect_ns=min_effect_ns)
    return StatisticsVerdict(
        k_repetitions=k_repetitions,
        block_count_used=block_count,
        verdict=verdict,
        ci_lower_ns=ci_lower_ns,
        ci_upper_ns=ci_upper_ns,
        min_effect_ns=min_effect_ns,
        baseline_reference_ns=baseline_reference_ns,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        confidence_bps=confidence_bps,
        calibration=calibration,
    )


__all__: Final = [
    "BASE_CONFIDENCE_BPS",
    "FORGED_TIMING",
    "FORGED_TIMING_MIN_ITERATION_NS",
    "FORGED_TIMING_TOLERANCE_NS",
    "STATISTICS_SCHEMA",
    "SampleTiming",
    "StatisticsVerdict",
    "VERDICTS",
    "VERDICT_IMPROVED",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_REGRESSED",
    "bca_bootstrap_ci",
    "bonferroni_confidence_bps",
    "classify_verdict",
    "compute_min_effect_ns",
    "compute_sample_timing",
    "compute_statistics_verdict",
    "max_peeks_for",
    "min_of_k",
    "parse_iteration_timings",
]
