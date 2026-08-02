"""Fail-closed dispatch from an exact-match activation decision."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_json, sha256_hex, strict_json_loads
from .contracts import Artifact, CandidateProposal, EvaluationPolicy, FrozenWorkload, RuntimeIdentity
from .errors import ContractError, FailureCode, UnknownFieldError
from .paths import validate_sha256
from .promotion import ACTIVATE, PromotionDecision
from .receipts import ContentAddressedStore, NATIVE_FALLBACK, Receipt, validate_receipt


DISPATCH_SCHEMA: Final = "auto_mlx.dispatch.v1"
NATIVE_MODE: Final = "native_fallback"
CANDIDATE_MODE: Final = "candidate"
DEFAULT_MAX_AGE_NS: Final = 24 * 60 * 60 * 1_000_000_000


def _exact(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if any(type(key) is not str for key in value):
        raise ContractError(f"{label} field names must be strings", code=FailureCode.WRONG_TYPE)
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise UnknownFieldError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ContractError(f"{label} is missing field(s): {', '.join(sorted(missing))}", code=FailureCode.INVALID_VALUE)


def _string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"{label} must be a non-empty string", code=FailureCode.WRONG_TYPE)
    return value


def _integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{label} must be a non-negative integer", code=FailureCode.WRONG_TYPE)
    return value


_DISPATCH_FIELDS: Final = {
    "schema",
    "mode",
    "reason",
    "candidate_id",
    "receipt_id",
    "decision_id",
    "created_at_ns",
    "dispatch_id",
}


@dataclass(frozen=True, slots=True, init=False)
class DispatchResult:
    """Closed result of the dispatch match, including native fallback."""

    mode: str
    reason: str
    candidate_id: str | None
    receipt_id: str | None
    decision_id: str | None
    created_at_ns: int
    dispatch_id: str

    def __init__(
        self,
        *,
        mode: str,
        reason: str,
        candidate_id: str | None,
        receipt_id: str | None,
        decision_id: str | None,
        created_at_ns: int | None = None,
        dispatch_id: str | None = None,
    ) -> None:
        mode = _string(mode, label="dispatch.mode")
        if mode not in {NATIVE_MODE, CANDIDATE_MODE}:
            raise ContractError("dispatch.mode is not closed", code=FailureCode.INVALID_VALUE)
        reason = _string(reason, label="dispatch.reason")
        if mode == NATIVE_MODE and any(value is not None for value in (candidate_id, receipt_id, decision_id)):
            raise ContractError("native fallback dispatch cannot carry activation identities", code=FailureCode.INVALID_VALUE)
        if mode == CANDIDATE_MODE and any(type(value) is not str or not value for value in (candidate_id, receipt_id, decision_id)):
            raise ContractError("candidate dispatch requires all activation identities", code=FailureCode.INVALID_VALUE)
        for label, value in (("candidate_id", candidate_id), ("receipt_id", receipt_id), ("decision_id", decision_id)):
            if value is not None:
                validate_sha256(value)
        timestamp = time.time_ns() if created_at_ns is None else _integer(created_at_ns, label="dispatch.created_at_ns")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "receipt_id", receipt_id)
        object.__setattr__(self, "decision_id", decision_id)
        object.__setattr__(self, "created_at_ns", timestamp)
        body = self._body_dict()
        expected_id = sha256_hex(body)
        if dispatch_id is not None:
            validate_sha256(dispatch_id)
            if dispatch_id != expected_id:
                raise ContractError("dispatch_id does not match canonical dispatch body", code=FailureCode.IDENTITY_MISMATCH)
        object.__setattr__(self, "dispatch_id", dispatch_id or expected_id)

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema": DISPATCH_SCHEMA,
            "mode": self.mode,
            "reason": self.reason,
            "candidate_id": self.candidate_id,
            "receipt_id": self.receipt_id,
            "decision_id": self.decision_id,
            "created_at_ns": self.created_at_ns,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._body_dict()
        result["dispatch_id"] = self.dispatch_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def native_fallback(self) -> bool:
        return self.mode == NATIVE_MODE

    @property
    def candidate(self) -> bool:
        return self.mode == CANDIDATE_MODE

    @classmethod
    def from_dict(cls, value: Any) -> "DispatchResult":
        if type(value) is not dict:
            raise ContractError("dispatch must be a JSON object", code=FailureCode.WRONG_TYPE)
        _exact(value, _DISPATCH_FIELDS, label="dispatch")
        if value["schema"] != DISPATCH_SCHEMA:
            raise ContractError("dispatch schema is incompatible", code=FailureCode.INVALID_VALUE)
        return cls(
            mode=value["mode"],
            reason=value["reason"],
            candidate_id=value["candidate_id"],
            receipt_id=value["receipt_id"],
            decision_id=value["decision_id"],
            created_at_ns=value["created_at_ns"],
            dispatch_id=value["dispatch_id"],
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "DispatchResult":
        return cls.from_dict(strict_json_loads(value))


Dispatch = DispatchResult


def _native(reason: str) -> DispatchResult:
    return DispatchResult(
        mode=NATIVE_MODE,
        reason=reason,
        candidate_id=None,
        receipt_id=None,
        decision_id=None,
    )


def _coerce_workload(value: FrozenWorkload | Mapping[str, Any]) -> FrozenWorkload:
    return value if isinstance(value, FrozenWorkload) else FrozenWorkload.from_dict(value)


def _coerce_policy(value: EvaluationPolicy | Mapping[str, Any]) -> EvaluationPolicy:
    return value if isinstance(value, EvaluationPolicy) else EvaluationPolicy.from_dict(value)


def _coerce_runtime(value: RuntimeIdentity | Mapping[str, Any]) -> RuntimeIdentity:
    return value if isinstance(value, RuntimeIdentity) else RuntimeIdentity.from_dict(value)


def _coerce_candidate(
    value: CandidateProposal | Mapping[str, Any], workload: FrozenWorkload
) -> CandidateProposal:
    return value if isinstance(value, CandidateProposal) else CandidateProposal.from_dict(value, workload)


def _coerce_artifacts(
    value: Sequence[Artifact | Mapping[str, Any]] | None, workload: FrozenWorkload
) -> tuple[Artifact, ...]:
    if value is None:
        return tuple(workload.artifacts)
    if type(value) not in {list, tuple}:
        raise ContractError("dispatch.artifacts must be an array", code=FailureCode.WRONG_TYPE)
    return tuple(item if isinstance(item, Artifact) else Artifact.from_dict(item) for item in value)


def dispatch(
    store: ContentAddressedStore,
    workload: FrozenWorkload | Mapping[str, Any],
    candidate: CandidateProposal | Mapping[str, Any],
    policy: EvaluationPolicy | Mapping[str, Any],
    runtime: RuntimeIdentity | Mapping[str, Any],
    *,
    artifacts: Sequence[Artifact | Mapping[str, Any]] | None = None,
    artifact_root: str | os.PathLike[str] | None = None,
    now_ns: int | None = None,
    max_age_ns: int | None = None,
    max_age_seconds: int | None = None,
    attestation_key: bytes | bytearray | None = None,
) -> DispatchResult:
    """Return a candidate only for an exact, fresh, independently valid match."""

    try:
        expected_workload = _coerce_workload(workload)
        expected_candidate = _coerce_candidate(candidate, expected_workload)
        expected_policy = _coerce_policy(policy)
        expected_runtime = _coerce_runtime(runtime)
        expected_artifacts = _coerce_artifacts(artifacts, expected_workload)
        if max_age_ns is not None:
            max_age = _integer(max_age_ns, label="max_age_ns")
        elif max_age_seconds is not None:
            max_age = _integer(max_age_seconds, label="max_age_seconds") * 1_000_000_000
        else:
            max_age = DEFAULT_MAX_AGE_NS
        current_time = time.time_ns() if now_ns is None else _integer(now_ns, label="now_ns")
        current_decision_id = store.current_decision_id()
        if current_decision_id == NATIVE_FALLBACK:
            return _native("native_fallback_pointer")
        decision = PromotionDecision.from_dict(store.get_decision(current_decision_id))
        if store.current_decision_id() != current_decision_id:
            return _native("activation_pointer_changed")
        if decision.decision_id != current_decision_id or decision.action != ACTIVATE or decision.source != "validated_local_receipt":
            return _native("decision_not_activatable")
        if decision.attestation is None:
            return _native("decision_attestation_missing")
        if decision.workload is None or decision.candidate is None or decision.policy is None or decision.runtime is None:
            return _native("decision_context_missing")
        if decision.workload.workload_hash != expected_workload.workload_hash:
            return _native("workload_mismatch")
        if decision.candidate.candidate_id != expected_candidate.candidate_id:
            return _native("candidate_mismatch")
        if decision.policy != expected_policy:
            return _native("policy_mismatch")
        if decision.runtime.identity != expected_runtime.identity:
            return _native("runtime_mismatch")
        if decision.artifacts != expected_artifacts:
            return _native("artifact_manifest_mismatch")
        if decision.receipt_id is None:
            return _native("receipt_identity_missing")
        receipt = store.get_receipt(decision.receipt_id)
        if receipt.receipt_id != decision.receipt_id:
            return _native("receipt_identity_mismatch")
        if receipt.workload.workload_hash != expected_workload.workload_hash:
            return _native("receipt_workload_mismatch")
        if receipt.candidate.candidate_id != expected_candidate.candidate_id:
            return _native("receipt_candidate_mismatch")
        if receipt.policy != expected_policy:
            return _native("receipt_policy_mismatch")
        if receipt.runtime.identity != expected_runtime.identity:
            return _native("receipt_runtime_mismatch")
        if receipt.artifacts != expected_artifacts:
            return _native("receipt_artifact_manifest_mismatch")
        if artifact_root is None:
            return _native("artifact_root_missing")
        validation = validate_receipt(
            receipt,
            workload=expected_workload,
            candidate=expected_candidate,
            policy=expected_policy,
            runtime=expected_runtime,
            artifact_root=artifact_root,
            attestation=decision.attestation,
            attestation_key=attestation_key,
        )
        if not validation.ok:
            return _native("receipt_validation_failed")
        if validation.attestation != decision.attestation or not validation.artifacts_verified:
            return _native("activation_proof_or_artifact_missing")
        gain = validation.recomputed.get("metrics", {}).get("gain", {})
        if not (
            gain.get("improved") is True
            and type(gain.get("delta_ns")) is int
            and gain["delta_ns"] > 0
            and type(gain.get("baseline_sum_ns")) is int
            and type(gain.get("candidate_sum_ns")) is int
            and gain["baseline_sum_ns"] > gain["candidate_sum_ns"]
            and gain.get("numerator") == gain["delta_ns"]
            and type(gain.get("denominator")) is int
            and gain["denominator"] > 0
        ):
            return _native("gain_not_positive")
        timestamps = (receipt.created_at_ns, decision.created_at_ns)
        if any(current_time < timestamp or current_time - timestamp > max_age for timestamp in timestamps):
            return _native("activation_stale")
        # The pointer is the mutable selector.  Re-read it after all
        # immutable objects and validation have been consumed so a concurrent
        # flip cannot return a candidate selected by an older generation.
        if store.current_decision_id() != current_decision_id:
            return _native("activation_pointer_changed")
        return DispatchResult(
            mode=CANDIDATE_MODE,
            reason="exact_match",
            candidate_id=expected_candidate.candidate_id,
            receipt_id=receipt.receipt_id,
            decision_id=decision.decision_id,
        )
    except Exception:
        # Dispatch is a safety boundary: malformed, missing, stale, or tampered
        # state never escapes as a candidate or as a storage exception.
        return _native("dispatch_state_invalid")


resolve_dispatch = dispatch
dispatch_candidate = dispatch


__all__: Final = [
    "CANDIDATE_MODE",
    "DEFAULT_MAX_AGE_NS",
    "DISPATCH_SCHEMA",
    "Dispatch",
    "DispatchResult",
    "NATIVE_MODE",
    "dispatch",
    "dispatch_candidate",
    "resolve_dispatch",
]
