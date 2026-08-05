"""G0 evaluator orchestration that returns observations only."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from .canonical import sha256_hex
from .contracts import CandidateProposal, EvaluationPolicy, RuntimeIdentity
from .errors import ContractError, FailureCode
from .executor import (
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionRecord,
    IsolationAuthority,
    IsolationProvider,
    TrustedRunnerRegistry,
    build_execution_plan,
)
from .measurement import MeasurementSample, PairedMeasurementBundle, PairedMeasurementPlan, assemble_measurement_bundle
from .oracle import ExactOutputOracle, OracleDescriptor, OracleResult
from .paths import validate_sha256
from .statistics import VERDICT_INCONCLUSIVE, compute_sample_timing, compute_statistics_verdict
from .thermal import thermal_preflight as _default_thermal_preflight


@dataclass(frozen=True, slots=True)
class Observation:
    sample_id: str
    arm: str
    record: ExecutionRecord
    oracle: OracleResult


def _observation_accepted(observation: Observation) -> bool:
    if type(observation) is not Observation or type(observation.record) is not ExecutionRecord or type(observation.oracle) is not OracleResult:
        return False
    record = observation.record
    return (
        record.status.value == "success"
        and record.parent_elapsed_ns > 0
        and record.failure is None
        and not record.stdout_truncated
        and not record.stderr_truncated
        and not record.output_truncated
        and observation.oracle.matched
        and observation.oracle.failure is None
    )


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    """Complete evaluator output with no promotion or dispatch capability."""

    candidate_id: str
    workload_hash: str
    runtime: RuntimeIdentity
    baseline_runner_id: str
    baseline_runner_digest: str
    candidate_runner_id: str
    candidate_runner_digest: str
    isolation_provider_id: str | None
    isolation_identity: str | None
    isolation_verifier_id: str | None
    isolation_verifier_identity: str | None
    warmups: tuple[Observation, ...]
    measurements: PairedMeasurementBundle
    thermal_blocks: tuple[Mapping[str, Any], ...] = ()
    isolation_requirements: frozenset[str] | None = None
    policy_digest: str | None = None
    execution_policy_digest: str | None = None
    measurement_block_count: int | None = None
    evaluation_policy: EvaluationPolicy | None = None
    execution_policy: ExecutionPolicy | None = None
    oracle: ExactOutputOracle | None = None
    oracle_descriptor: OracleDescriptor | None = None
    # Wave B: the sequential BCa-bootstrap verdict computed over
    # ``measurements`` (see auto_mlx.statistics.compute_statistics_verdict).
    # ``None`` only when ``measurements`` never reached an accepted state
    # (no valid evidence to compute a verdict from) -- see ``evaluate()``.
    statistics: Mapping[str, Any] | None = None

    def _canonical_measurement_plan(self) -> PairedMeasurementPlan:
        policy = self.evaluation_policy
        execution_policy = self.execution_policy
        if type(policy) is not EvaluationPolicy or type(execution_policy) is not ExecutionPolicy:
            raise ContractError("observation bundle policy material is missing", code=FailureCode.INVALID_POLICY)
        if self.policy_digest != sha256_hex(policy.to_dict()):
            raise ContractError("observation bundle policy digest is not bound", code=FailureCode.IDENTITY_MISMATCH)
        expected_execution_policy = _execution_policy_from_contract(policy)
        if execution_policy.to_dict() != expected_execution_policy.to_dict():
            raise ContractError("observation bundle execution policy is not bound", code=FailureCode.INVALID_POLICY)
        if self.execution_policy_digest != _execution_policy_digest(execution_policy):
            raise ContractError("observation bundle execution policy digest is not bound", code=FailureCode.IDENTITY_MISMATCH)
        # The actual block count varies with Wave B sequential extension
        # (policy.measurement_runs is the starting/minimum count,
        # policy.max_measurement_runs the cap); any count in that closed
        # range is legitimate here -- the stronger "did it stop for a good
        # reason" proof (decisive verdict, or hit the cap) lives with the
        # statistics verdict itself, not this identity binding.
        if (
            type(self.measurement_block_count) is not int
            or not (policy.measurement_runs <= self.measurement_block_count <= policy.max_measurement_runs)
        ):
            raise ContractError("observation bundle block count is not bound", code=FailureCode.INVALID_POLICY)
        if type(self.oracle) is not ExactOutputOracle or type(self.oracle_descriptor) is not OracleDescriptor:
            raise ContractError("observation bundle oracle provenance is missing", code=FailureCode.ORACLE_MISMATCH)
        if self.oracle.descriptor != self.oracle_descriptor:
            raise ContractError("observation bundle oracle provenance is mismatched", code=FailureCode.IDENTITY_MISMATCH)
        if not self._bundle_identity_is_bound(execution_policy):
            raise ContractError("observation bundle identity is not bound", code=FailureCode.IDENTITY_MISMATCH)
        return PairedMeasurementPlan.create(
            self.measurement_block_count,
            candidate_id=self.candidate_id,
            workload_hash=self.workload_hash,
            baseline_runner_id=self.baseline_runner_id,
            baseline_runner_digest=self.baseline_runner_digest,
            candidate_runner_id=self.candidate_runner_id,
            candidate_runner_digest=self.candidate_runner_digest,
            oracle=self.oracle,
            require_isolation=True,
            isolation_provider_id=self.isolation_provider_id,
            isolation_identity=self.isolation_identity,
            isolation_verifier_id=self.isolation_verifier_id,
            isolation_verifier_identity=self.isolation_verifier_identity,
            isolation_requirements=self.isolation_requirements,
        )

    def recompute_measurements(self) -> PairedMeasurementBundle:
        """Recompute measurement acceptance from canonical slots and raw records."""

        if type(self.measurements) is not PairedMeasurementBundle:
            raise ContractError("observation bundle measurements are missing", code=FailureCode.WRONG_TYPE)
        plan = self._canonical_measurement_plan()
        if self.measurements.plan_digest != plan.plan_digest:
            raise ContractError("observation bundle measurement plan is not bound", code=FailureCode.IDENTITY_MISMATCH)
        recomputed = assemble_measurement_bundle(plan, self.measurements.raw_samples)
        # Block/sample structure is evidence.  Top-level accepted/rejection
        # booleans are caller-supplied summaries and are intentionally ignored.
        if recomputed.blocks != self.measurements.blocks:
            raise ContractError("observation bundle measurement structure is inconsistent", code=FailureCode.IDENTITY_MISMATCH)
        return recomputed

    def _warmups_are_bound(self) -> bool:
        policy = self.evaluation_policy
        oracle = self.oracle
        if type(policy) is not EvaluationPolicy or type(oracle) is not ExactOutputOracle:
            return False
        if type(self.warmups) is not tuple or len(self.warmups) != policy.warmup_runs * 2:
            return False
        for index, observation in enumerate(self.warmups):
            expected_arm = "baseline" if index % 2 == 0 else "candidate"
            expected_id = f"warmup-{index // 2 + 1:04d}-{expected_arm}"
            expected_runner_id = self.baseline_runner_id if expected_arm == "baseline" else self.candidate_runner_id
            expected_runner_digest = self.baseline_runner_digest if expected_arm == "baseline" else self.candidate_runner_digest
            record = observation.record if type(observation) is Observation else None
            if (
                type(observation) is not Observation
                or observation.sample_id != expected_id
                or observation.arm != expected_arm
                or type(record) is not ExecutionRecord
                or record.observation_id != expected_id
                or record.arm != expected_arm
                or record.candidate_id != self.candidate_id
                or record.workload_hash != self.workload_hash
                or record.runner_id != expected_runner_id
                or record.runner_digest != expected_runner_digest
                or record.isolation is None
                or record.isolation.provider_id != self.isolation_provider_id
                or record.isolation.identity != self.isolation_identity
                or record.isolation.verifier_id != self.isolation_verifier_id
                or record.isolation.verifier_identity != self.isolation_verifier_identity
                or self.isolation_requirements is None
                or not self.isolation_requirements.issubset(record.isolation.requirements)
                or observation.oracle != oracle.evaluate(record.stdout)
                or not _observation_accepted(observation)
            ):
                return False
        return True

    @property
    def accepted(self) -> bool:
        """Whether every warmup and measurement sample is genuine, bound evidence.

        Computed from real evidence -- warmup identity/isolation/oracle
        binding plus a from-scratch recomputation of the measurement bundle
        -- never from a caller-supplied summary.  Absent or malformed
        evidence (unbound identity, missing isolation, forged provenance)
        fails closed to False, exactly as before; this only stops being
        unconditionally False once genuine evidence supports it.
        """

        if not self._warmups_are_bound():
            return False
        try:
            measurements = self.recompute_measurements()
        except ContractError:
            return False
        return measurements.accepted

    @property
    def promotion_eligible(self) -> bool:
        """Whether this bundle is complete, verified-isolation evidence.

        This is the G1 evidence layer's own judgment -- it is a necessary
        input to, but not the same thing as, G2 production activation
        (receipt validation and promotion decisions independently
        recompute and gate on their own evidence; see
        docs/evidence-and-promotion.md).
        """

        if not self._warmups_are_bound():
            return False
        try:
            measurements = self.recompute_measurements()
        except ContractError:
            return False
        return measurements.promotion_eligible

    def _bundle_identity_is_bound(self, execution_policy: ExecutionPolicy) -> bool:
        if type(self.candidate_id) is not str or type(self.workload_hash) is not str:
            return False
        validate_sha256(self.candidate_id)
        validate_sha256(self.workload_hash)
        if type(self.runtime) is not RuntimeIdentity:
            return False
        for value in (self.baseline_runner_id, self.candidate_runner_id):
            if type(value) is not str or not value:
                return False
        for value in (self.baseline_runner_digest, self.candidate_runner_digest):
            validate_sha256(value)
        isolation_values = (
            self.isolation_provider_id,
            self.isolation_identity,
            self.isolation_verifier_id,
            self.isolation_verifier_identity,
            self.isolation_requirements,
        )
        if all(value is None for value in isolation_values):
            return True
        if any(value is None for value in isolation_values):
            return False
        if type(self.isolation_provider_id) is not str or not self.isolation_provider_id:
            return False
        if type(self.isolation_verifier_id) is not str or not self.isolation_verifier_id:
            return False
        validate_sha256(self.isolation_identity)
        validate_sha256(self.isolation_verifier_identity)
        if type(self.isolation_requirements) is not frozenset:
            return False
        return self.isolation_requirements == execution_policy.required_isolation

    @property
    def paired_measurements(self) -> PairedMeasurementBundle:
        return self.measurements

    @property
    def raw_records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(item.record for item in self.warmups) + self.measurements.raw_records

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "workload_hash": self.workload_hash,
            "runtime": self.runtime.to_dict(),
            "baseline_runner_id": self.baseline_runner_id,
            "baseline_runner_digest": self.baseline_runner_digest,
            "candidate_runner_id": self.candidate_runner_id,
            "candidate_runner_digest": self.candidate_runner_digest,
            "isolation_provider_id": self.isolation_provider_id,
            "isolation_identity": self.isolation_identity,
            "isolation_verifier_id": self.isolation_verifier_id,
            "isolation_verifier_identity": self.isolation_verifier_identity,
            "isolation_requirements": sorted(self.isolation_requirements) if self.isolation_requirements is not None else None,
            "policy_digest": self.policy_digest,
            "execution_policy_digest": self.execution_policy_digest,
            "measurement_block_count": self.measurement_block_count,
            "evaluation_policy": self.evaluation_policy.to_dict() if self.evaluation_policy is not None else None,
            "execution_policy": self.execution_policy.to_dict() if self.execution_policy is not None else None,
            "oracle_descriptor": self.oracle_descriptor.to_dict() if self.oracle_descriptor is not None else None,
            "promotion_eligible": self.promotion_eligible,
            "warmups": [
                {
                    "sample_id": item.sample_id,
                    "arm": item.arm,
                    "status": item.record.status.value,
                    "oracle_matched": item.oracle.matched,
                    "parent_elapsed_ns": item.record.parent_elapsed_ns,
                    "runner_elapsed_ns": item.record.runner_elapsed_ns,
                }
                for item in self.warmups
            ],
            "measurements": self.measurements.to_dict(),
            "thermal_blocks": [dict(item) for item in self.thermal_blocks],
            "statistics": None if self.statistics is None else dict(self.statistics),
        }


def _execution_policy_from_contract(policy: EvaluationPolicy) -> ExecutionPolicy:
    return ExecutionPolicy(
        timeout_seconds=float(policy.timeout_seconds),
        max_stdout_bytes=policy.max_output_bytes,
        max_stderr_bytes=policy.max_output_bytes,
        max_output_bytes=policy.max_output_bytes,
        require_network_denial=True,
        require_descendant_containment=True,
        # Wave B: tells the runner subprocess how many in-runner timed
        # iterations to perform (see auto_mlx.runners.reference_matmul and
        # auto_mlx.statistics).  Part of ExecutionPolicy.to_dict(), so
        # changing k_repetitions changes the execution policy digest --
        # correctly binding K into the evaluation's identity chain.
        extra_environment={"AUTO_MLX_K_REPETITIONS": str(policy.k_repetitions)},
    )


def _execution_policy_digest(policy: ExecutionPolicy) -> str:
    """Bind numeric execution settings without introducing JSON floats."""

    values = policy.to_dict()
    for name in ("timeout_seconds", "kill_grace_seconds", "launch_timeout_seconds", "authority_timeout_seconds", "reader_join_timeout_seconds"):
        values[name] = format(values[name], ".17g")
    return sha256_hex(values)


def _block_point_estimates(measurements: PairedMeasurementBundle) -> tuple[list[list[int]], list[list[int]]]:
    """Per-block, per-arm point estimates for an ACCEPTED measurement bundle.

    Requires ``measurements.accepted`` (every block's four samples carry a
    successful record) -- callers must check that first.  Each sample's
    point estimate is computed purely from its own evidence (see
    ``auto_mlx.statistics.compute_sample_timing``): trusted min-of-K when
    the runner's self-reported iteration array survives the forged-timing
    cross-check, otherwise the parent-observed ``runner_elapsed_ns`` alone.
    """

    baseline_points: list[list[int]] = []
    candidate_points: list[list[int]] = []
    for block in measurements.blocks:
        block_baseline: list[int] = []
        block_candidate: list[int] = []
        for sample in block.samples:
            record = sample.record
            if record is None:
                raise ContractError(
                    "cannot compute statistics from an incomplete measurement block",
                    code=FailureCode.RUNTIME_FAILURE,
                )
            timing = compute_sample_timing(record.runner_elapsed_ns, record.stderr)
            (block_baseline if sample.arm == "baseline" else block_candidate).append(timing.point_estimate_ns)
        baseline_points.append(block_baseline)
        candidate_points.append(block_candidate)
    return baseline_points, candidate_points


class Evaluator:
    """Evaluate a declarative proposal using evaluator-owned runners and isolation."""

    def __init__(
        self,
        registry: TrustedRunnerRegistry,
        *,
        baseline_runner_id: str,
        candidate_runner_id: str,
        oracle: ExactOutputOracle,
        artifact_root: str,
        policy: EvaluationPolicy | None = None,
        execution_policy: ExecutionPolicy | None = None,
        provider: IsolationProvider | None = None,
        authority: IsolationAuthority | None = None,
        block_count: int | None = None,
        thermal_preflight: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(registry, TrustedRunnerRegistry):
            raise ContractError("evaluator requires a TrustedRunnerRegistry", code=FailureCode.PROVIDER_ERROR)
        if not isinstance(oracle, ExactOutputOracle):
            raise ContractError("evaluator requires an evaluator-owned ExactOutputOracle", code=FailureCode.WRONG_TYPE)
        if type(artifact_root) is not str or not artifact_root:
            raise ContractError("artifact_root must be a non-empty string", code=FailureCode.WRONG_TYPE)
        if policy is not None and not isinstance(policy, EvaluationPolicy):
            raise ContractError("policy must be an EvaluationPolicy", code=FailureCode.WRONG_TYPE)
        if execution_policy is not None and not isinstance(execution_policy, ExecutionPolicy):
            raise ContractError("execution_policy must be an ExecutionPolicy", code=FailureCode.WRONG_TYPE)
        if provider is not None and not isinstance(provider, IsolationProvider):
            raise ContractError("provider must be an evaluator-owned IsolationProvider", code=FailureCode.SANDBOX_UNAVAILABLE)
        if authority is not None and not isinstance(authority, IsolationAuthority):
            raise ContractError("authority must be an evaluator-owned IsolationAuthority", code=FailureCode.SANDBOX_UNAVAILABLE)
        if block_count is not None and (type(block_count) is not int or block_count <= 0):
            raise ContractError("block_count must be a positive integer", code=FailureCode.INVALID_POLICY)
        if thermal_preflight is not None and not callable(thermal_preflight):
            raise ContractError("thermal_preflight must be callable", code=FailureCode.WRONG_TYPE)
        self._policy = policy or EvaluationPolicy()
        expected_execution_policy = _execution_policy_from_contract(self._policy)
        self._execution_policy = execution_policy or expected_execution_policy
        if self._execution_policy.to_dict() != expected_execution_policy.to_dict():
            raise ContractError(
                "execution_policy must exactly match the declared evaluation policy",
                code=FailureCode.INVALID_POLICY,
            )
        if block_count is not None and block_count != self._policy.measurement_runs:
            raise ContractError(
                "block_count must exactly match policy.measurement_runs",
                code=FailureCode.INVALID_POLICY,
            )
        self._registry = registry
        self._baseline_runner_id = baseline_runner_id
        self._candidate_runner_id = candidate_runner_id
        self._oracle = oracle
        self._artifact_root = artifact_root
        self._block_count = self._policy.measurement_runs
        self._policy_digest = sha256_hex(self._policy.to_dict())
        self._execution_policy_digest = _execution_policy_digest(self._execution_policy)
        self._baseline_runner = registry.resolve(baseline_runner_id)
        self._candidate_runner = registry.resolve(candidate_runner_id)
        # Held only to be forwarded into plan.execute(); evaluate() never
        # reads a provider/authority's own identity properties directly
        # (see test_evaluate_never_reads_external_isolation_metadata) --
        # bundle-level isolation identity is derived exclusively from the
        # VerifiedIsolation a real execution actually returns.
        self._provider = provider
        self._authority = authority
        # Defaults to a real pmset -g therm preflight with a real 30s retry
        # sleep (see auto_mlx.thermal.thermal_preflight); tests inject a
        # fast, canned callable instead of waiting on real hardware/timing.
        self._thermal_preflight = thermal_preflight or _default_thermal_preflight

    def _run(self, plan: ExecutionPlan, sample_id: str, arm: str) -> Observation:
        record = plan.execute(
            self._execution_policy,
            registry=self._registry,
            provider=self._provider,
            authority=self._authority,
            observation_id=sample_id,
            arm=arm,
        )
        return Observation(sample_id, arm, record, self._oracle.evaluate(record.stdout))

    def evaluate(self, proposal: CandidateProposal) -> ObservationBundle:
        if not isinstance(proposal, CandidateProposal):
            raise ContractError("evaluator accepts only CandidateProposal values", code=FailureCode.WRONG_TYPE)
        baseline_plan = build_execution_plan(proposal, self._registry, self._baseline_runner_id, self._artifact_root)
        candidate_plan = build_execution_plan(proposal, self._registry, self._candidate_runner_id, self._artifact_root)
        warmups: list[Observation] = []
        for index in range(self._policy.warmup_runs):
            warmups.append(self._run(baseline_plan, f"warmup-{index + 1:04d}-baseline", "baseline"))
            warmups.append(self._run(candidate_plan, f"warmup-{index + 1:04d}-candidate", "candidate"))

        # Isolation identity is derived exclusively from evidence a real
        # execution actually returned (the first warmup's VerifiedIsolation,
        # if any) -- never by reading provider/authority properties
        # directly.  Absent real evidence, identity stays fully unbound and
        # every downstream record is later rejected for missing isolation.
        first_isolation = warmups[0].record.isolation if warmups else None
        if first_isolation is not None:
            isolation_provider_id = first_isolation.provider_id
            isolation_identity = first_isolation.identity
            isolation_verifier_id = first_isolation.verifier_id
            isolation_verifier_identity = first_isolation.verifier_identity
            isolation_requirements = self._execution_policy.required_isolation
        else:
            isolation_provider_id = None
            isolation_identity = None
            isolation_verifier_id = None
            isolation_verifier_identity = None
            isolation_requirements = None

        # Wave B sequential sampling: start at policy.measurement_runs
        # blocks; while the bootstrap verdict stays inconclusive, extend one
        # block at a time up to policy.max_measurement_runs, recomputing the
        # verdict after every new block and stopping early on a decisive
        # one.  Extending rebuilds the measurement plan at the new, larger
        # block count -- block identity (block_id/sequence) only depends on
        # an index, so every previously-run block's samples stay valid slot
        # bindings and are never re-executed (see executed_block_ids).  The
        # bootstrap seed is drawn once, up front, and reused unchanged at
        # every peek so the whole procedure -- and any later independent
        # recomputation of it -- is a pure function of (seed, resamples,
        # differences-at-that-peek).
        count = self._block_count
        bootstrap_seed = secrets.randbits(63)
        samples: list[MeasurementSample] = []
        thermal_blocks: list[dict[str, Any]] = []
        executed_block_ids: set[str] = set()
        measurement_plan: PairedMeasurementPlan
        measurements: PairedMeasurementBundle
        statistics_result: Mapping[str, Any] | None = None

        while True:
            measurement_plan = PairedMeasurementPlan.create(
                count,
                candidate_id=proposal.candidate_id,
                workload_hash=proposal.workload_hash,
                baseline_runner_id=baseline_plan.runner_id,
                baseline_runner_digest=baseline_plan.runner_digest,
                candidate_runner_id=candidate_plan.runner_id,
                candidate_runner_digest=candidate_plan.runner_digest,
                oracle=self._oracle,
                require_isolation=True,
                isolation_provider_id=isolation_provider_id,
                isolation_identity=isolation_identity,
                isolation_verifier_id=isolation_verifier_id,
                isolation_verifier_identity=isolation_verifier_identity,
                isolation_requirements=isolation_requirements,
            )
            for block in measurement_plan.blocks:
                if block.block_id in executed_block_ids:
                    continue
                # Preflight before this block's own samples, not once for
                # the whole evaluate() call -- thermal state can change
                # between blocks over a multi-minute evaluation.
                preflight = dict(self._thermal_preflight())
                thermally_suspect = bool(preflight.get("thermally_suspect"))
                refused = thermally_suspect and self._policy.thermal_gate_policy == "refuse"
                thermal_blocks.append(
                    {
                        "block_id": block.block_id,
                        "block_index": block.block_index,
                        "policy": self._policy.thermal_gate_policy,
                        "preflight": preflight,
                        "refused": refused,
                    }
                )
                executed_block_ids.add(block.block_id)
                if refused:
                    # Skip this block's samples entirely; the measurement
                    # bundle's existing missing-sample handling rejects the
                    # block the same way it would for any other incomplete
                    # block -- no separate rejection code needed.
                    continue
                for slot in block.slots:
                    execution_plan = baseline_plan if slot.arm == "baseline" else candidate_plan
                    observation = self._run(execution_plan, slot.sample_id, slot.arm)
                    samples.append(
                        MeasurementSample(
                            sample_id=slot.sample_id,
                            block_id=slot.block_id,
                            slot_index=slot.slot_index,
                            arm=slot.arm,
                            record=observation.record,
                            oracle=observation.oracle,
                        )
                    )
            measurements = assemble_measurement_bundle(measurement_plan, samples)
            if not measurements.accepted:
                # No valid, complete evidence to compute a verdict from --
                # stop extending; the bundle surfaces as unaccepted/failed
                # exactly as before Wave B, just without a statistics field.
                statistics_result = None
                break
            baseline_points, candidate_points = _block_point_estimates(measurements)
            verdict = compute_statistics_verdict(
                block_baseline_points=baseline_points,
                block_candidate_points=candidate_points,
                k_repetitions=self._policy.k_repetitions,
                measurement_runs=self._policy.measurement_runs,
                max_measurement_runs=self._policy.max_measurement_runs,
                min_effect_bps=self._policy.min_effect_bps,
                bootstrap_resamples=self._policy.bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
                calibration=self._policy.calibration,
            )
            statistics_result = verdict.to_dict()
            if verdict.verdict != VERDICT_INCONCLUSIVE or count >= self._policy.max_measurement_runs:
                break
            count += 1

        return ObservationBundle(
            candidate_id=proposal.candidate_id,
            workload_hash=proposal.workload_hash,
            runtime=RuntimeIdentity.current(),
            baseline_runner_id=baseline_plan.runner_id,
            baseline_runner_digest=baseline_plan.runner_digest,
            candidate_runner_id=candidate_plan.runner_id,
            candidate_runner_digest=candidate_plan.runner_digest,
            isolation_provider_id=isolation_provider_id,
            isolation_identity=isolation_identity,
            isolation_verifier_id=isolation_verifier_id,
            isolation_verifier_identity=isolation_verifier_identity,
            warmups=tuple(warmups),
            measurements=measurements,
            thermal_blocks=tuple(thermal_blocks),
            isolation_requirements=isolation_requirements,
            policy_digest=self._policy_digest,
            execution_policy_digest=self._execution_policy_digest,
            measurement_block_count=count,
            evaluation_policy=self._policy,
            execution_policy=self._execution_policy,
            oracle=self._oracle,
            oracle_descriptor=self._oracle.descriptor,
            statistics=statistics_result,
        )


__all__: Final = ["Evaluator", "Observation", "ObservationBundle"]
