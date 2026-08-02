"""G0 evaluator orchestration that returns observations only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

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
from .oracle import ExactOutputOracle, OracleResult


@dataclass(frozen=True, slots=True)
class Observation:
    sample_id: str
    arm: str
    record: ExecutionRecord
    oracle: OracleResult


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
    isolation_requirements: frozenset[str] | None = None

    @property
    def accepted(self) -> bool:
        return self.measurements.accepted

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
            "warmups": [
                {
                    "sample_id": item.sample_id,
                    "arm": item.arm,
                    "status": item.record.status.value,
                    "oracle_matched": item.oracle.matched,
                    "parent_elapsed_ns": item.record.parent_elapsed_ns,
                }
                for item in self.warmups
            ],
            "measurements": self.measurements.to_dict(),
        }


def _execution_policy_from_contract(policy: EvaluationPolicy) -> ExecutionPolicy:
    return ExecutionPolicy(
        timeout_seconds=float(policy.timeout_seconds),
        max_stdout_bytes=policy.max_output_bytes,
        max_stderr_bytes=policy.max_output_bytes,
        max_output_bytes=policy.max_output_bytes,
        require_network_denial=True,
        require_descendant_containment=True,
    )


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
        self._registry = registry
        self._baseline_runner_id = baseline_runner_id
        self._candidate_runner_id = candidate_runner_id
        self._oracle = oracle
        self._artifact_root = artifact_root
        self._policy = policy or EvaluationPolicy()
        self._execution_policy = execution_policy or _execution_policy_from_contract(self._policy)
        self._provider = provider
        self._authority = authority
        self._block_count = block_count
        self._baseline_runner = registry.resolve(baseline_runner_id)
        self._candidate_runner = registry.resolve(candidate_runner_id)

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

        count = self._block_count if self._block_count is not None else self._policy.measurement_runs
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
            isolation_provider_id=self._provider.provider_id if self._provider else None,
            isolation_identity=self._provider.identity if self._provider else None,
            isolation_verifier_id=self._authority.verifier_id if self._authority else None,
            isolation_verifier_identity=self._authority.identity if self._authority else None,
            isolation_requirements=self._execution_policy.required_isolation if self._provider and self._authority else None,
        )
        samples: list[MeasurementSample] = []
        for block in measurement_plan.blocks:
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
        return ObservationBundle(
            candidate_id=proposal.candidate_id,
            workload_hash=proposal.workload_hash,
            runtime=RuntimeIdentity.current(),
            baseline_runner_id=baseline_plan.runner_id,
            baseline_runner_digest=baseline_plan.runner_digest,
            candidate_runner_id=candidate_plan.runner_id,
            candidate_runner_digest=candidate_plan.runner_digest,
            isolation_provider_id=self._provider.provider_id if self._provider else None,
            isolation_identity=self._provider.identity if self._provider else None,
            isolation_verifier_id=self._authority.verifier_id if self._authority else None,
            isolation_verifier_identity=self._authority.identity if self._authority else None,
            warmups=tuple(warmups),
            measurements=measurements,
            isolation_requirements=self._execution_policy.required_isolation if self._provider and self._authority else None,
        )


__all__: Final = ["Evaluator", "Observation", "ObservationBundle"]
