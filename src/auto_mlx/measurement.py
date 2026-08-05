"""Complete deterministic paired timing blocks for G0 observations.

Assembly is an independent evidence check: it re-runs the evaluator-owned
exact oracle over every raw stdout value and validates all runner/workload/
candidate/slot provenance before accepting a block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from .canonical import sha256_hex
from .errors import AutoMLXError, ContractError, FailureCode
from .executor import ExecutionRecord, ExecutionStatus, VerifiedIsolation
from .oracle import ExactOutputOracle, OracleResult
from .paths import validate_sha256


Arm = Literal["baseline", "candidate"]
_BASELINE: Final = "baseline"
_CANDIDATE: Final = "candidate"
_REQUIRED_ISOLATION: Final = frozenset({"network_denial", "descendant_containment"})
_PRODUCTION_ISOLATION_UNAVAILABLE: Final = "production_isolation_unavailable"


def _arm(value: Any) -> Arm:
    if value not in {_BASELINE, _CANDIDATE}:
        raise ContractError("measurement arm must be baseline or candidate", code=FailureCode.WRONG_TYPE)
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class MeasurementSlot:
    sample_id: str
    block_id: str
    slot_index: int
    arm: Arm


@dataclass(frozen=True, slots=True)
class PairedBlock:
    block_id: str
    block_index: int
    sequence: tuple[Arm, Arm, Arm, Arm]
    slots: tuple[MeasurementSlot, MeasurementSlot, MeasurementSlot, MeasurementSlot]

    def __post_init__(self) -> None:
        if self.sequence not in {
            (_BASELINE, _CANDIDATE, _CANDIDATE, _BASELINE),
            (_CANDIDATE, _BASELINE, _BASELINE, _CANDIDATE),
        }:
            raise ContractError("paired block must be ABBA or BAAB", code=FailureCode.INVALID_VALUE)
        if len(self.slots) != 4 or tuple(slot.arm for slot in self.slots) != self.sequence:
            raise ContractError("paired block slots must match its sequence", code=FailureCode.INVALID_VALUE)


def _measurement_plan_digest(
    blocks: tuple[PairedBlock, ...],
    *,
    candidate_id: str | None,
    workload_hash: str | None,
    baseline_runner_id: str | None,
    baseline_runner_digest: str | None,
    candidate_runner_id: str | None,
    candidate_runner_digest: str | None,
    oracle: ExactOutputOracle | None,
    require_isolation: bool,
    isolation_provider_id: str | None,
    isolation_identity: str | None,
    isolation_verifier_id: str | None,
    isolation_verifier_identity: str | None,
    isolation_requirements: frozenset[str] | None,
) -> str:
    return sha256_hex(
        {
            "block_count": len(blocks),
            "blocks": [
                {
                    "block_id": block.block_id,
                    "block_index": block.block_index,
                    "sequence": list(block.sequence),
                    "slots": [
                        {
                            "sample_id": slot.sample_id,
                            "block_id": slot.block_id,
                            "slot_index": slot.slot_index,
                            "arm": slot.arm,
                        }
                        for slot in block.slots
                    ],
                }
                for block in blocks
            ],
            "candidate_id": candidate_id,
            "workload_hash": workload_hash,
            "baseline_runner_id": baseline_runner_id,
            "baseline_runner_digest": baseline_runner_digest,
            "candidate_runner_id": candidate_runner_id,
            "candidate_runner_digest": candidate_runner_digest,
            "oracle_digest": oracle.expected_digest if oracle else None,
            "require_isolation": require_isolation,
            "isolation_provider_id": isolation_provider_id,
            "isolation_identity": isolation_identity,
            "isolation_verifier_id": isolation_verifier_id,
            "isolation_verifier_identity": isolation_verifier_identity,
            "isolation_requirements": sorted(isolation_requirements) if isolation_requirements is not None else None,
        }
    )


@dataclass(frozen=True, slots=True)
class PairedMeasurementPlan:
    blocks: tuple[PairedBlock, ...]
    plan_digest: str
    candidate_id: str | None = None
    workload_hash: str | None = None
    baseline_runner_id: str | None = None
    baseline_runner_digest: str | None = None
    candidate_runner_id: str | None = None
    candidate_runner_digest: str | None = None
    oracle: ExactOutputOracle | None = None
    require_isolation: bool = True
    isolation_provider_id: str | None = None
    isolation_identity: str | None = None
    isolation_verifier_id: str | None = None
    isolation_verifier_identity: str | None = None
    isolation_requirements: frozenset[str] | None = None

    def __post_init__(self) -> None:
        validate_sha256(self.plan_digest)
        if type(self.blocks) is not tuple or not self.blocks:
            raise ContractError("measurement plan blocks must be a non-empty tuple", code=FailureCode.INVALID_POLICY)
        for index, block in enumerate(self.blocks):
            expected_sequence = (
                (_BASELINE, _CANDIDATE, _CANDIDATE, _BASELINE)
                if index % 2 == 0
                else (_CANDIDATE, _BASELINE, _BASELINE, _CANDIDATE)
            )
            expected_block_id = f"block-{index + 1:04d}"
            if (
                not isinstance(block, PairedBlock)
                or block.block_index != index
                or block.block_id != expected_block_id
                or block.sequence != expected_sequence
                or tuple(slot.block_id for slot in block.slots) != (expected_block_id,) * 4
                or tuple(slot.slot_index for slot in block.slots) != (0, 1, 2, 3)
                or tuple(slot.sample_id for slot in block.slots)
                != tuple(f"{expected_block_id}-slot-{slot_index + 1}" for slot_index in range(4))
            ):
                raise ContractError("measurement plan sequence or slot identity is not canonical", code=FailureCode.IDENTITY_MISMATCH)
        expected_digest = _measurement_plan_digest(
            self.blocks,
            candidate_id=self.candidate_id,
            workload_hash=self.workload_hash,
            baseline_runner_id=self.baseline_runner_id,
            baseline_runner_digest=self.baseline_runner_digest,
            candidate_runner_id=self.candidate_runner_id,
            candidate_runner_digest=self.candidate_runner_digest,
            oracle=self.oracle,
            require_isolation=self.require_isolation,
            isolation_provider_id=self.isolation_provider_id,
            isolation_identity=self.isolation_identity,
            isolation_verifier_id=self.isolation_verifier_id,
            isolation_verifier_identity=self.isolation_verifier_identity,
            isolation_requirements=self.isolation_requirements,
        )
        if self.plan_digest != expected_digest:
            raise ContractError("measurement plan digest does not match its immutable fields", code=FailureCode.IDENTITY_MISMATCH)

    @classmethod
    def create(
        cls,
        block_count: int,
        *,
        candidate_id: str | None = None,
        workload_hash: str | None = None,
        baseline_runner_id: str | None = None,
        baseline_runner_digest: str | None = None,
        candidate_runner_id: str | None = None,
        candidate_runner_digest: str | None = None,
        oracle: ExactOutputOracle | None = None,
        require_isolation: bool = True,
        isolation_provider_id: str | None = None,
        isolation_identity: str | None = None,
        isolation_verifier_id: str | None = None,
        isolation_verifier_identity: str | None = None,
        isolation_requirements: frozenset[str] | None = None,
    ) -> "PairedMeasurementPlan":
        if type(block_count) is not int or block_count <= 0:
            raise ContractError("block_count must be a positive integer", code=FailureCode.INVALID_POLICY)
        if type(require_isolation) is not bool:
            raise ContractError("require_isolation must be a boolean", code=FailureCode.INVALID_POLICY)
        if isolation_provider_id is not None and (type(isolation_provider_id) is not str or not isolation_provider_id):
            raise ContractError("isolation_provider_id must be a non-empty string or null", code=FailureCode.WRONG_TYPE)
        if isolation_identity is not None:
            validate_sha256(isolation_identity)
        if isolation_verifier_id is not None and (type(isolation_verifier_id) is not str or not isolation_verifier_id):
            raise ContractError("isolation_verifier_id must be a non-empty string or null", code=FailureCode.WRONG_TYPE)
        if isolation_verifier_identity is not None:
            validate_sha256(isolation_verifier_identity)
        if isolation_requirements is not None:
            if type(isolation_requirements) not in {set, frozenset} or any(
                type(item) is not str or not item for item in isolation_requirements
            ):
                raise ContractError("isolation_requirements must be a string set or null", code=FailureCode.WRONG_TYPE)
            isolation_requirements = frozenset(isolation_requirements)
        isolation_values = (
            isolation_provider_id,
            isolation_identity,
            isolation_verifier_id,
            isolation_verifier_identity,
            isolation_requirements,
        )
        if any(value is not None for value in isolation_values) and not all(value is not None for value in isolation_values):
            raise ContractError("isolation provider provenance must be complete or absent", code=FailureCode.INVALID_VALUE)
        if require_isolation and isolation_requirements is not None and not _REQUIRED_ISOLATION.issubset(isolation_requirements):
            raise ContractError("measurement isolation requirements must include network and descendant enforcement", code=FailureCode.INVALID_POLICY)
        if not require_isolation and any(value is not None for value in isolation_values):
            raise ContractError("isolation provenance cannot be bound when isolation is not required", code=FailureCode.INVALID_POLICY)
        for name, value in (
            ("candidate_id", candidate_id),
            ("workload_hash", workload_hash),
            ("baseline_runner_id", baseline_runner_id),
            ("candidate_runner_id", candidate_runner_id),
        ):
            if value is not None and (type(value) is not str or not value):
                raise ContractError(f"{name} must be a non-empty string or null", code=FailureCode.WRONG_TYPE)
        for name, value in (("candidate_id", candidate_id), ("workload_hash", workload_hash)):
            if value is not None:
                validate_sha256(value)
        for name, value in (("baseline_runner_digest", baseline_runner_digest), ("candidate_runner_digest", candidate_runner_digest)):
            if value is not None and (type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
                raise ContractError(f"{name} must be a lowercase digest or null", code=FailureCode.WRONG_TYPE)
        if oracle is not None and not isinstance(oracle, ExactOutputOracle):
            raise ContractError("oracle must be an ExactOutputOracle or null", code=FailureCode.WRONG_TYPE)
        blocks: list[PairedBlock] = []
        for index in range(block_count):
            sequence: tuple[Arm, Arm, Arm, Arm]
            if index % 2 == 0:
                sequence = (_BASELINE, _CANDIDATE, _CANDIDATE, _BASELINE)
            else:
                sequence = (_CANDIDATE, _BASELINE, _BASELINE, _CANDIDATE)
            block_id = f"block-{index + 1:04d}"
            slots = tuple(
                MeasurementSlot(
                    sample_id=f"{block_id}-slot-{slot_index + 1}",
                    block_id=block_id,
                    slot_index=slot_index,
                    arm=arm,
                )
                for slot_index, arm in enumerate(sequence)
            )
            blocks.append(PairedBlock(block_id, index, sequence, slots))
        digest = _measurement_plan_digest(
            tuple(blocks),
            candidate_id=candidate_id,
            workload_hash=workload_hash,
            baseline_runner_id=baseline_runner_id,
            baseline_runner_digest=baseline_runner_digest,
            candidate_runner_id=candidate_runner_id,
            candidate_runner_digest=candidate_runner_digest,
            oracle=oracle,
            require_isolation=require_isolation,
            isolation_provider_id=isolation_provider_id,
            isolation_identity=isolation_identity,
            isolation_verifier_id=isolation_verifier_id,
            isolation_verifier_identity=isolation_verifier_identity,
            isolation_requirements=isolation_requirements,
        )
        return cls(
            blocks=tuple(blocks),
            plan_digest=digest,
            candidate_id=candidate_id,
            workload_hash=workload_hash,
            baseline_runner_id=baseline_runner_id,
            baseline_runner_digest=baseline_runner_digest,
            candidate_runner_id=candidate_runner_id,
            candidate_runner_digest=candidate_runner_digest,
            oracle=oracle,
            require_isolation=require_isolation,
            isolation_provider_id=isolation_provider_id,
            isolation_identity=isolation_identity,
            isolation_verifier_id=isolation_verifier_id,
            isolation_verifier_identity=isolation_verifier_identity,
            isolation_requirements=isolation_requirements,
        )

    @property
    def expected_sample_ids(self) -> tuple[str, ...]:
        return tuple(slot.sample_id for block in self.blocks for slot in block.slots)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def bound(self) -> bool:
        try:
            self.__post_init__()
        except (ContractError, TypeError, ValueError):
            return False
        identity_bound = all(
            value is not None
            for value in (
                self.candidate_id,
                self.workload_hash,
                self.baseline_runner_id,
                self.baseline_runner_digest,
                self.candidate_runner_id,
                self.candidate_runner_digest,
                self.oracle,
            )
        )
        isolation_bound = (
            not self.require_isolation
            or (
                self.isolation_provider_id is not None
                and self.isolation_identity is not None
                and self.isolation_verifier_id is not None
                and self.isolation_verifier_identity is not None
                and self.isolation_requirements is not None
                and _REQUIRED_ISOLATION.issubset(self.isolation_requirements)
            )
        )
        return identity_bound and isolation_bound


@dataclass(frozen=True, slots=True)
class MeasurementSample:
    sample_id: str
    block_id: str
    slot_index: int
    arm: Arm
    record: ExecutionRecord | None
    oracle: OracleResult | None

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise ContractError("sample_id must be a non-empty string", code=FailureCode.WRONG_TYPE)
        if type(self.block_id) is not str or not self.block_id:
            raise ContractError("block_id must be a non-empty string", code=FailureCode.WRONG_TYPE)
        if type(self.slot_index) is not int or self.slot_index < 0:
            raise ContractError("slot_index must be a non-negative integer", code=FailureCode.WRONG_TYPE)
        _arm(self.arm)
        if self.record is not None and type(self.record) is not ExecutionRecord:
            raise ContractError("measurement record must be an ExecutionRecord or null", code=FailureCode.WRONG_TYPE)
        if self.oracle is not None and not isinstance(self.oracle, OracleResult):
            raise ContractError("measurement oracle must be an OracleResult or null", code=FailureCode.WRONG_TYPE)

    @property
    def missing(self) -> bool:
        return self.record is None


@dataclass(frozen=True, slots=True)
class DispersionInputs:
    """Raw timing vectors only; no universal estimator or pass threshold.

    ``*_parent_elapsed_ns`` is the full-sample span (staging, launch, wait,
    and authority verification) and is retained for diagnostics only.
    ``*_runner_elapsed_ns`` is the evidentiary quantity -- the runner
    subprocess's own launch-to-exit span, excluding verification probe time
    (see ``execute_plan`` in ``auto_mlx.executor``) -- and is what all
    downstream measurement/gain math must read.
    """

    ordered_parent_elapsed_ns: tuple[int | None, ...]
    baseline_elapsed_ns: tuple[int | None, ...]
    candidate_elapsed_ns: tuple[int | None, ...]
    ordered_runner_elapsed_ns: tuple[int | None, ...] = ()
    baseline_runner_elapsed_ns: tuple[int | None, ...] = ()
    candidate_runner_elapsed_ns: tuple[int | None, ...] = ()

    @property
    def baseline_drift_ns(self) -> int | None:
        if len(self.baseline_elapsed_ns) != 2 or any(value is None for value in self.baseline_elapsed_ns):
            return None
        return self.baseline_elapsed_ns[1] - self.baseline_elapsed_ns[0]  # type: ignore[operator]


@dataclass(frozen=True, slots=True)
class BlockObservation:
    block: PairedBlock
    samples: tuple[MeasurementSample, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    dispersion_inputs: DispersionInputs

    @property
    def baseline_drift_ns(self) -> int | None:
        return self.dispersion_inputs.baseline_drift_ns

    @property
    def raw_records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(sample.record for sample in self.samples if sample.record is not None)


class MeasurementRejected(AutoMLXError):
    code = FailureCode.RUNTIME_FAILURE


@dataclass(frozen=True, slots=True)
class PairedMeasurementBundle:
    plan_digest: str
    blocks: tuple[BlockObservation, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]
    unexpected_sample_ids: tuple[str, ...] = ()

    @property
    def raw_samples(self) -> tuple[MeasurementSample, ...]:
        return tuple(sample for block in self.blocks for sample in block.samples)

    @property
    def raw_records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(sample.record for sample in self.raw_samples if sample.record is not None)

    @property
    def promotion_eligible(self) -> bool:
        """Structurally complete, verified-isolation, correctness-matched evidence.

        This requires everything ``accepted`` requires (complete paired
        samples, oracle match, no runtime failures) *and* that every raw
        record carries verified isolation evidence.  The extra isolation
        check matters when ``require_isolation`` was False on the plan:
        ``accepted`` can be True for non-isolated raw measurements (kept for
        diagnosis), but they are never promotion eligible.
        """

        return self.accepted and all(record.isolation is not None for record in self.raw_records)

    def require_complete(self) -> "PairedMeasurementBundle":
        if not self.accepted:
            raise MeasurementRejected("paired measurement bundle rejected: " + "; ".join(self.rejection_reasons))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_digest": self.plan_digest,
            "accepted": self.accepted,
            "promotion_eligible": self.promotion_eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "unexpected_sample_ids": list(self.unexpected_sample_ids),
            "blocks": [
                {
                    "block_id": observation.block.block_id,
                    "sequence": list(observation.block.sequence),
                    "accepted": observation.accepted,
                    "rejection_reasons": list(observation.rejection_reasons),
                    "samples": [
                        {
                            "sample_id": sample.sample_id,
                            "arm": sample.arm,
                            "status": sample.record.status.value if sample.record else None,
                            "oracle_matched": sample.oracle.matched if sample.oracle else None,
                            "parent_elapsed_ns": sample.record.parent_elapsed_ns if sample.record else None,
                            "runner_elapsed_ns": sample.record.runner_elapsed_ns if sample.record else None,
                            "isolation": bool(sample.record and sample.record.isolation),
                        }
                        for sample in observation.samples
                    ],
                    "dispersion_inputs": {
                        "ordered_parent_elapsed_ns": list(observation.dispersion_inputs.ordered_parent_elapsed_ns),
                        "baseline_elapsed_ns": list(observation.dispersion_inputs.baseline_elapsed_ns),
                        "candidate_elapsed_ns": list(observation.dispersion_inputs.candidate_elapsed_ns),
                        "ordered_runner_elapsed_ns": list(observation.dispersion_inputs.ordered_runner_elapsed_ns),
                        "baseline_runner_elapsed_ns": list(observation.dispersion_inputs.baseline_runner_elapsed_ns),
                        "candidate_runner_elapsed_ns": list(observation.dispersion_inputs.candidate_runner_elapsed_ns),
                        "baseline_drift_ns": observation.baseline_drift_ns,
                    },
                }
                for observation in self.blocks
            ],
        }


def _sample_map(samples: Mapping[str, MeasurementSample] | Sequence[MeasurementSample]) -> tuple[dict[str, MeasurementSample], tuple[str, ...]]:
    if isinstance(samples, Mapping):
        values = tuple(samples.values())
    elif type(samples) in {tuple, list}:
        values = tuple(samples)
    else:
        raise ContractError("measurement samples must be a mapping or array", code=FailureCode.WRONG_TYPE)
    result: dict[str, MeasurementSample] = {}
    duplicates: list[str] = []
    for sample in values:
        if not isinstance(sample, MeasurementSample):
            raise ContractError("measurement samples must contain MeasurementSample values", code=FailureCode.WRONG_TYPE)
        if sample.sample_id in result:
            duplicates.append(sample.sample_id)
        result[sample.sample_id] = sample
    return result, tuple(duplicates)


def _expected_runner(plan: PairedMeasurementPlan, arm: Arm) -> tuple[str, str]:
    if arm == _BASELINE:
        assert plan.baseline_runner_id is not None and plan.baseline_runner_digest is not None
        return plan.baseline_runner_id, plan.baseline_runner_digest
    assert plan.candidate_runner_id is not None and plan.candidate_runner_digest is not None
    return plan.candidate_runner_id, plan.candidate_runner_digest


def assemble_measurement_bundle(
    plan: PairedMeasurementPlan,
    samples: Mapping[str, MeasurementSample] | Sequence[MeasurementSample],
) -> PairedMeasurementBundle:
    """Reject incomplete, failed, forged, or provenance-mismatched blocks."""

    if not isinstance(plan, PairedMeasurementPlan):
        raise ContractError("measurement assembly requires a PairedMeasurementPlan", code=FailureCode.WRONG_TYPE)
    sample_map, duplicate_ids = _sample_map(samples)
    expected_ids = set(plan.expected_sample_ids)
    unexpected = set(sample_map) - expected_ids
    isolation_provenance_bound = all(
        value is not None
        for value in (
            plan.isolation_provider_id,
            plan.isolation_identity,
            plan.isolation_verifier_id,
            plan.isolation_verifier_identity,
            plan.isolation_requirements,
        )
    )
    bundle_reasons: list[str] = []
    if not plan.bound:
        bundle_reasons.append("unbound_plan")
    if plan.require_isolation and not isolation_provenance_bound:
        # The plan itself never bound isolation evidence, so production
        # isolation is structurally unavailable regardless of any per-sample
        # evidence.  When provenance IS bound, per-slot checks below
        # (isolation_unverified/isolation_provider_mismatch/...)
        # independently catch any real evidence problem; this reason is not
        # appended a second time on top of them.
        bundle_reasons.append(_PRODUCTION_ISOLATION_UNAVAILABLE)
        bundle_reasons.append("unbound_isolation_provenance")
    if duplicate_ids:
        bundle_reasons.append("duplicate_sample_ids:" + ",".join(sorted(set(duplicate_ids))))
    if unexpected:
        bundle_reasons.append("unexpected_sample_ids:" + ",".join(sorted(unexpected)))

    observations: list[BlockObservation] = []
    for block in plan.blocks:
        block_samples: list[MeasurementSample] = []
        reasons: list[str] = (
            [_PRODUCTION_ISOLATION_UNAVAILABLE] if plan.require_isolation and not isolation_provenance_bound else []
        )
        for slot in block.slots:
            sample = sample_map.get(slot.sample_id)
            if sample is None:
                sample = MeasurementSample(slot.sample_id, slot.block_id, slot.slot_index, slot.arm, None, None)
                reasons.append(f"missing:{slot.sample_id}")
            else:
                if (sample.block_id, sample.slot_index, sample.arm) != (slot.block_id, slot.slot_index, slot.arm):
                    reasons.append(f"slot_mismatch:{slot.sample_id}")
                if sample.record is None:
                    reasons.append(f"missing_record:{slot.sample_id}")
                else:
                    expected_runner_id, expected_runner_digest = _expected_runner(plan, slot.arm) if plan.bound else (None, None)
                    if plan.candidate_id is None or sample.record.candidate_id != plan.candidate_id:
                        reasons.append(f"candidate_identity_mismatch:{slot.sample_id}")
                    if plan.workload_hash is None or sample.record.workload_hash != plan.workload_hash:
                        reasons.append(f"workload_identity_mismatch:{slot.sample_id}")
                    if expected_runner_id is None or sample.record.runner_id != expected_runner_id or sample.record.runner_digest != expected_runner_digest:
                        reasons.append(f"runner_identity_mismatch:{slot.sample_id}")
                    if sample.record.observation_id != slot.sample_id or sample.record.arm != slot.arm:
                        reasons.append(f"record_slot_identity_mismatch:{slot.sample_id}")
                    if plan.require_isolation:
                        if sample.record.isolation is None:
                            reasons.append(f"isolation_unverified:{slot.sample_id}")
                        elif type(sample.record.isolation) is not VerifiedIsolation:
                            reasons.append(f"isolation_type_mismatch:{slot.sample_id}")
                        else:
                            if (
                                sample.record.isolation.provider_id != plan.isolation_provider_id
                                or sample.record.isolation.identity != plan.isolation_identity
                                or sample.record.isolation.verifier_id != plan.isolation_verifier_id
                                or sample.record.isolation.verifier_identity != plan.isolation_verifier_identity
                            ):
                                reasons.append(f"isolation_provider_mismatch:{slot.sample_id}")
                            if (
                                plan.isolation_requirements is None
                                or not plan.isolation_requirements.issubset(sample.record.isolation.requirements)
                            ):
                                reasons.append(f"isolation_requirements_mismatch:{slot.sample_id}")
                    if sample.record.status is not ExecutionStatus.SUCCESS:
                        reasons.append(f"execution_failure:{slot.sample_id}:{sample.record.status.value}")
                    if sample.record.parent_elapsed_ns <= 0:
                        reasons.append(f"zero_duration:{slot.sample_id}")
                    if sample.record.runner_elapsed_ns is None or sample.record.runner_elapsed_ns <= 0:
                        reasons.append(f"zero_runner_duration:{slot.sample_id}")
                    if (
                        sample.record.stdout_truncated
                        or sample.record.stderr_truncated
                        or sample.record.output_truncated
                    ):
                        reasons.append(f"truncated_output:{slot.sample_id}")
                    if sample.record.failure is not None:
                        reasons.append(f"record_failure_metadata:{slot.sample_id}")
                    if plan.oracle is not None:
                        recomputed = plan.oracle.evaluate(sample.record.stdout)
                        if sample.oracle is None:
                            reasons.append(f"missing_oracle_metadata:{slot.sample_id}")
                        elif sample.oracle != recomputed:
                            reasons.append(f"forged_oracle_metadata:{slot.sample_id}")
                        if sample.oracle is not None and sample.oracle.matched and sample.oracle.failure is not None:
                            reasons.append(f"oracle_failure_metadata:{slot.sample_id}")
                        if not recomputed.matched:
                            reasons.append(f"oracle_mismatch:{slot.sample_id}")
                    else:
                        reasons.append(f"oracle_unbound:{slot.sample_id}")
            block_samples.append(sample)
        ordered = tuple(sample.record.parent_elapsed_ns if sample.record else None for sample in block_samples)
        baseline = tuple(sample.record.parent_elapsed_ns if sample.record else None for sample in block_samples if sample.arm == _BASELINE)
        candidate = tuple(sample.record.parent_elapsed_ns if sample.record else None for sample in block_samples if sample.arm == _CANDIDATE)
        ordered_runner = tuple(sample.record.runner_elapsed_ns if sample.record else None for sample in block_samples)
        baseline_runner = tuple(
            sample.record.runner_elapsed_ns if sample.record else None for sample in block_samples if sample.arm == _BASELINE
        )
        candidate_runner = tuple(
            sample.record.runner_elapsed_ns if sample.record else None for sample in block_samples if sample.arm == _CANDIDATE
        )
        observation = BlockObservation(
            block=block,
            samples=tuple(block_samples),
            accepted=not reasons and plan.bound,
            rejection_reasons=tuple(reasons),
            dispersion_inputs=DispersionInputs(ordered, baseline, candidate, ordered_runner, baseline_runner, candidate_runner),
        )
        observations.append(observation)
        bundle_reasons.extend(reasons)
    if len(sample_map) != len(expected_ids):
        bundle_reasons.append("incomplete_sample_set")
    accepted = not bundle_reasons and all(observation.accepted for observation in observations)
    return PairedMeasurementBundle(
        plan_digest=plan.plan_digest,
        blocks=tuple(observations),
        accepted=accepted,
        rejection_reasons=tuple(dict.fromkeys(bundle_reasons)),
        unexpected_sample_ids=tuple(sorted(unexpected)),
    )


assemble_paired_measurements = assemble_measurement_bundle


__all__: Final = [
    "Arm",
    "BlockObservation",
    "DispersionInputs",
    "MeasurementRejected",
    "MeasurementSample",
    "MeasurementSlot",
    "PairedBlock",
    "PairedMeasurementBundle",
    "PairedMeasurementPlan",
    "assemble_measurement_bundle",
    "assemble_paired_measurements",
]
