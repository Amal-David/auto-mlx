"""Shared Wave B test fixture: a hand-built, genuinely evaluator-bundle-backed receipt.

Promotion and dispatch now gate on the independently recomputed Wave B
statistics verdict (see ``auto_mlx.promotion``/``auto_mlx.dispatch``), which
only exists for evaluator-bundle-backed receipts -- the older flat
raw-sample lane has no paired-block/per-iteration evidence to derive a
verdict from, so it is (correctly, per Wave B's fail-closed contract)
never promotable.  Tests that exercise promotion/dispatch mechanics
(activation, tampering, HMAC recompute, staleness, ...) need a receipt that
can genuinely activate, so they build one through this shared helper rather
than each hand-rolling the evaluator-bundle wire.

This mirrors ``tests/test_receipts.py``'s own fixtures but is deliberately
deterministic: every block gets the identical, trusted min-of-K point
estimate for its arm (a single reported iteration per sample, with
``runner_elapsed_ns`` set to match it exactly), so the BCa bootstrap CI
collapses to a zero-width point at exactly
``baseline_iteration_ns - candidate_iteration_ns`` regardless of block
count, resample count, or seed.
"""

from __future__ import annotations

import json
from typing import Any

from auto_mlx.canonical import sha256_hex
from auto_mlx.contracts import CandidateProposal, EvaluationPolicy, FrozenWorkload, RuntimeIdentity
from auto_mlx.evaluator import Observation, ObservationBundle, _execution_policy_digest, _execution_policy_from_contract
from auto_mlx.executor import ExecutionRecord, ExecutionStatus, IsolationAuthority, IsolationClaim, IsolatedProcess, IsolationProvider
from auto_mlx.measurement import MeasurementSample, PairedMeasurementPlan, assemble_measurement_bundle
from auto_mlx.oracle import ExactOutputOracle
from auto_mlx.receipts import Receipt
from auto_mlx.runners.reference_matmul import ITER_TIMINGS_MARKER, WARMUP_MARKER
from auto_mlx.statistics import compute_sample_timing, compute_statistics_verdict
from auto_mlx.thermal import ThermalReading


class FixtureIsolationProvider(IsolationProvider):
    def __init__(self) -> None:
        super().__init__("wave-b-fixture-isolation", "1" * 64)

    def enforce(self, argv: Any, **kwargs: Any) -> IsolatedProcess:  # pragma: no cover - never invoked
        raise NotImplementedError("this fixture only ever hand-builds ExecutionRecord values")


class FixtureIsolationAuthority(IsolationAuthority):
    def __init__(self) -> None:
        super().__init__("wave-b-fixture-verifier", "2" * 64, production_eligible=False)

    def verify(self, provider: IsolationProvider, process: Any, claim: IsolationClaim):
        return self._attest(provider, claim)


def _iter_timings_stderr(iteration_ns: int) -> bytes:
    payload = json.dumps({"k": 1, "iterations_ns": [iteration_ns]}, separators=(",", ":"))
    return f"{WARMUP_MARKER}\n{ITER_TIMINGS_MARKER} {payload}\n".encode("ascii")


def _nominal_thermal_blocks(plan: PairedMeasurementPlan, *, policy: EvaluationPolicy) -> tuple[dict[str, Any], ...]:
    reading = ThermalReading("nominal", 100, None, None, "").to_dict()
    preflight = {"initial": reading, "final": reading, "retried": False, "thermally_suspect": False}
    return tuple(
        {
            "block_id": block.block_id,
            "block_index": block.block_index,
            "policy": policy.thermal_gate_policy,
            "preflight": preflight,
            "refused": False,
        }
        for block in plan.blocks
    )


def build_evaluator_bundle_receipt(
    workload: FrozenWorkload,
    candidate: CandidateProposal,
    policy: EvaluationPolicy,
    runtime: RuntimeIdentity,
    *,
    baseline_iteration_ns: int = 20_000_000,
    candidate_iteration_ns: int = 10_000_000,
    bootstrap_seed: int = 1,
    created_at_ns: int = 100,
) -> Receipt:
    """A structurally complete, accepted, evaluator-bundle-backed receipt.

    Defaults to a candidate ~2x faster than baseline (a decisive
    ``improved`` verdict against the 2% default threshold); pass
    ``baseline_iteration_ns < candidate_iteration_ns`` for a decisive
    ``regressed`` fixture instead.  Raises ``AssertionError`` (a test
    fixture failure, not a production error path) if the resulting bundle
    is not accepted -- callers should never see that in practice given
    these deterministic, trusted inputs.
    """

    oracle = ExactOutputOracle(b"ok\n", label="wave-b-fixture-oracle")
    provider = FixtureIsolationProvider()
    authority = FixtureIsolationAuthority()
    plan = PairedMeasurementPlan.create(
        policy.measurement_runs,
        candidate_id=candidate.candidate_id,
        workload_hash=workload.workload_hash,
        baseline_runner_id="baseline",
        baseline_runner_digest="1" * 64,
        candidate_runner_id="candidate",
        candidate_runner_digest="2" * 64,
        oracle=oracle,
        isolation_provider_id=provider.provider_id,
        isolation_identity=provider.identity,
        isolation_verifier_id=authority.verifier_id,
        isolation_verifier_identity=authority.identity,
        isolation_requirements=frozenset({"network_denial", "descendant_containment"}),
    )

    def record(sample_id: str, arm: str) -> ExecutionRecord:
        runner_id = "baseline" if arm == "baseline" else "candidate"
        runner_digest = "1" * 64 if arm == "baseline" else "2" * 64
        iteration_ns = baseline_iteration_ns if arm == "baseline" else candidate_iteration_ns
        return ExecutionRecord(
            candidate_id=candidate.candidate_id,
            workload_hash=workload.workload_hash,
            runner_id=runner_id,
            runner_digest=runner_digest,
            status=ExecutionStatus.SUCCESS,
            parent_elapsed_ns=iteration_ns + 1_000,
            runner_elapsed_ns=iteration_ns,
            observation_id=sample_id,
            arm=arm,
            returncode=0,
            stdout=b"ok\n",
            stderr=_iter_timings_stderr(iteration_ns),
            isolation=authority._attest(provider, provider._claim("d" * 64)),
        )

    warmups = tuple(
        Observation(
            f"warmup-{index + 1:04d}-{arm}",
            arm,
            record(f"warmup-{index + 1:04d}-{arm}", arm),
            oracle.evaluate(b"ok\n"),
        )
        for index in range(policy.warmup_runs)
        for arm in ("baseline", "candidate")
    )
    samples = []
    for block in plan.blocks:
        for slot in block.slots:
            current = record(slot.sample_id, slot.arm)
            samples.append(
                MeasurementSample(slot.sample_id, slot.block_id, slot.slot_index, slot.arm, current, oracle.evaluate(current.stdout))
            )
    measurements = assemble_measurement_bundle(plan, samples)
    if not measurements.accepted:
        raise AssertionError(f"fixture measurement bundle was not accepted: {measurements.rejection_reasons}")

    baseline_points: list[list[int]] = []
    candidate_points: list[list[int]] = []
    for block in measurements.blocks:
        block_baseline: list[int] = []
        block_candidate: list[int] = []
        for sample in block.samples:
            timing = compute_sample_timing(sample.record.runner_elapsed_ns, sample.record.stderr)
            (block_baseline if sample.arm == "baseline" else block_candidate).append(timing.point_estimate_ns)
        baseline_points.append(block_baseline)
        candidate_points.append(block_candidate)
    verdict = compute_statistics_verdict(
        block_baseline_points=baseline_points,
        block_candidate_points=candidate_points,
        k_repetitions=policy.k_repetitions,
        measurement_runs=policy.measurement_runs,
        max_measurement_runs=policy.max_measurement_runs,
        min_effect_bps=policy.min_effect_bps,
        bootstrap_resamples=policy.bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        calibration=policy.calibration,
    )

    bundle = ObservationBundle(
        candidate_id=candidate.candidate_id,
        workload_hash=workload.workload_hash,
        runtime=runtime,
        baseline_runner_id="baseline",
        baseline_runner_digest="1" * 64,
        candidate_runner_id="candidate",
        candidate_runner_digest="2" * 64,
        isolation_provider_id=provider.provider_id,
        isolation_identity=provider.identity,
        isolation_verifier_id=authority.verifier_id,
        isolation_verifier_identity=authority.identity,
        isolation_requirements=frozenset({"network_denial", "descendant_containment"}),
        warmups=warmups,
        measurements=measurements,
        thermal_blocks=_nominal_thermal_blocks(plan, policy=policy),
        policy_digest=sha256_hex(policy.to_dict()),
        execution_policy_digest=_execution_policy_digest(_execution_policy_from_contract(policy)),
        measurement_block_count=policy.measurement_runs,
        evaluation_policy=policy,
        execution_policy=_execution_policy_from_contract(policy),
        oracle=oracle,
        oracle_descriptor=oracle.descriptor,
        statistics=verdict.to_dict(),
    )
    if not bundle.accepted or not bundle.promotion_eligible:
        raise AssertionError("fixture bundle was not accepted/promotion-eligible")

    return Receipt.from_observation_bundle(bundle, workload, candidate, policy, oracle=oracle, created_at_ns=created_at_ns)
