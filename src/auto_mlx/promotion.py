"""Pure promotion decisions for validated local receipts.

Promotion never consumes an evaluator recommendation or a mutable observation.
It accepts only :class:`ReceiptValidation`, then emits an immutable decision
whose public and performance claims remain explicitly withheld.
"""

from __future__ import annotations

import os
import time
import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from .canonical import canonical_bytes, canonical_json, sha256_hex, strict_json_loads, validate_json_value
from .contracts import Artifact, CandidateProposal, EvaluationPolicy, FrozenWorkload, RuntimeIdentity
from .errors import ContractError, Failure, FailureCode, UnknownFieldError
from .paths import validate_sha256
from .receipts import (
    CLAIMS_WITHHELD,
    NATIVE_FALLBACK,
    ContentAddressedStore,
    Receipt,
    ReceiptValidation,
    receipt_attestation,
    validate_receipt,
)


DECISION_SCHEMA: Final = "auto_mlx.decision.v1"
ACTIVATE: Final = "activate"
NATIVE: Final = "native_fallback"


def _object(value: Any, *, label: str) -> dict[str, Any]:
    validate_json_value(value)
    if type(value) is not dict:
        raise ContractError(f"{label} must be a JSON object", code=FailureCode.WRONG_TYPE)
    return value


def _exact(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if any(type(key) is not str for key in value):
        raise ContractError(f"{label} field names must be strings", code=FailureCode.WRONG_TYPE)
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise UnknownFieldError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ContractError(
            f"{label} is missing field(s): {', '.join(sorted(missing))}",
            code=FailureCode.INVALID_VALUE,
        )


def _string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"{label} must be a non-empty string", code=FailureCode.WRONG_TYPE)
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ContractError(f"{label} must not contain unpaired surrogates", code=FailureCode.INVALID_UNICODE)
    return value


def _integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{label} must be a non-negative integer", code=FailureCode.WRONG_TYPE)
    return value


def _claims() -> dict[str, str]:
    return {"public": CLAIMS_WITHHELD, "performance": CLAIMS_WITHHELD}


def _coerce_context(
    receipt: Receipt,
    *,
    workload: FrozenWorkload | None = None,
    candidate: CandidateProposal | None = None,
    policy: EvaluationPolicy | None = None,
    runtime: RuntimeIdentity | None = None,
    artifacts: Sequence[Artifact] | None = None,
) -> tuple[FrozenWorkload, CandidateProposal, EvaluationPolicy, RuntimeIdentity, tuple[Artifact, ...]]:
    return (
        workload or receipt.workload,
        candidate or receipt.candidate,
        policy or receipt.policy,
        runtime or receipt.runtime,
        tuple(receipt.artifacts if artifacts is None else artifacts),
    )


_DECISION_FIELDS: Final = {
    "schema",
    "source",
    "action",
    "reason",
    "receipt_id",
    "workload",
    "candidate",
    "policy",
    "runtime",
    "artifacts",
    "created_at_ns",
    "claims",
    "attestation",
    "decision_id",
}


@dataclass(frozen=True, slots=True, init=False)
class PromotionDecision:
    """A closed immutable activation or native-fallback decision."""

    source: str
    action: str
    reason: str
    receipt_id: str | None
    workload: FrozenWorkload | None
    candidate: CandidateProposal | None
    policy: EvaluationPolicy | None
    runtime: RuntimeIdentity | None
    artifacts: tuple[Artifact, ...]
    created_at_ns: int
    claims: Mapping[str, str]
    attestation: str | None
    decision_id: str

    def __init__(
        self,
        *,
        source: str,
        action: str,
        reason: str,
        receipt_id: str | None,
        workload: FrozenWorkload | None,
        candidate: CandidateProposal | None,
        policy: EvaluationPolicy | None,
        runtime: RuntimeIdentity | None,
        artifacts: Sequence[Artifact] = (),
        created_at_ns: int | None = None,
        claims: Mapping[str, str] | None = None,
        attestation: str | None = None,
        decision_id: str | None = None,
    ) -> None:
        source = _string(source, label="decision.source")
        action = _string(action, label="decision.action")
        reason = _string(reason, label="decision.reason")
        if action not in {ACTIVATE, NATIVE}:
            raise ContractError("decision.action must be activate or native_fallback", code=FailureCode.INVALID_VALUE)
        if source not in {"validated_local_receipt", "promotion_rejected", "rollback"}:
            raise ContractError("decision.source is not an allowed promotion source", code=FailureCode.PROMOTION_REJECTED)
        if action == ACTIVATE and source != "validated_local_receipt":
            raise ContractError("only a validated local receipt can activate", code=FailureCode.PROMOTION_REJECTED)
        if action == NATIVE and source not in {"promotion_rejected", "rollback"}:
            raise ContractError("native fallback decisions require a rejection or rollback source", code=FailureCode.PROMOTION_REJECTED)
        if source == "rollback" and (receipt_id is not None or workload is not None or candidate is not None or policy is not None or runtime is not None or artifacts):
            raise ContractError("rollback decisions cannot carry candidate activation context", code=FailureCode.PROMOTION_REJECTED)
        if action == ACTIVATE:
            if not isinstance(workload, FrozenWorkload) or not isinstance(candidate, CandidateProposal):
                raise ContractError("activation decision requires workload and candidate", code=FailureCode.PROMOTION_REJECTED)
            if not isinstance(policy, EvaluationPolicy) or not isinstance(runtime, RuntimeIdentity):
                raise ContractError("activation decision requires policy and runtime", code=FailureCode.PROMOTION_REJECTED)
            if type(receipt_id) is not str:
                raise ContractError("activation decision requires receipt_id", code=FailureCode.PROMOTION_REJECTED)
            validate_sha256(receipt_id)
            if candidate.workload_hash != workload.workload_hash:
                raise ContractError("decision candidate does not belong to workload", code=FailureCode.IDENTITY_MISMATCH)
            if type(attestation) is not str:
                raise ContractError("activation decision requires evaluator attestation", code=FailureCode.PROMOTION_REJECTED)
            validate_sha256(attestation)
        elif attestation is not None:
            raise ContractError("fallback decisions cannot carry attestation", code=FailureCode.PROMOTION_REJECTED)
        elif receipt_id is not None:
            validate_sha256(receipt_id)
        if type(artifacts) not in {list, tuple}:
            raise ContractError("decision.artifacts must be an array", code=FailureCode.WRONG_TYPE)
        frozen_artifacts = tuple(artifacts)
        if any(not isinstance(artifact, Artifact) for artifact in frozen_artifacts):
            raise ContractError("decision.artifacts must contain Artifact values", code=FailureCode.WRONG_TYPE)
        timestamp = time.time_ns() if created_at_ns is None else _integer(created_at_ns, label="decision.created_at_ns")
        if claims is None:
            final_claims = _claims()
        else:
            if type(claims) is not dict:
                raise ContractError("decision.claims must be an object", code=FailureCode.WRONG_TYPE)
            _claim_keys = {"public", "performance"}
            if set(claims) != _claim_keys:
                raise ContractError("decision.claims must contain exactly public and performance", code=FailureCode.INVALID_VALUE)
            if any(type(value) is not str for value in claims.values()):
                raise ContractError("decision claim values must be strings", code=FailureCode.WRONG_TYPE)
            final_claims = dict(claims)
        if final_claims != _claims():
            raise ContractError("decision claims must remain withheld pending external attestation", code=FailureCode.PROMOTION_REJECTED)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "receipt_id", receipt_id)
        object.__setattr__(self, "workload", workload)
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "artifacts", frozen_artifacts)
        object.__setattr__(self, "created_at_ns", timestamp)
        object.__setattr__(self, "claims", MappingProxyType(dict(final_claims)))
        object.__setattr__(self, "attestation", attestation)
        expected_id = sha256_hex(self._body_dict())
        if decision_id is None:
            final_id = expected_id
        else:
            final_id = validate_sha256(decision_id)
            if final_id != expected_id:
                raise ContractError("decision_id does not match canonical decision body", code=FailureCode.IDENTITY_MISMATCH)
        object.__setattr__(self, "decision_id", final_id)

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema": DECISION_SCHEMA,
            "source": self.source,
            "action": self.action,
            "reason": self.reason,
            "receipt_id": self.receipt_id,
            "workload": self.workload.to_dict() if self.workload is not None else None,
            "candidate": self.candidate.to_dict() if self.candidate is not None else None,
            "policy": self.policy.to_dict() if self.policy is not None else None,
            "runtime": self.runtime.to_dict() if self.runtime is not None else None,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "created_at_ns": self.created_at_ns,
            "claims": dict(self.claims),
            "attestation": self.attestation,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._body_dict()
        result["decision_id"] = self.decision_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def identity(self) -> str:
        return self.decision_id

    @property
    def activates(self) -> bool:
        return self.action == ACTIVATE

    @classmethod
    def from_dict(cls, value: Any) -> "PromotionDecision":
        data = _object(value, label="promotion decision")
        _exact(data, _DECISION_FIELDS, label="promotion decision")
        if data["schema"] != DECISION_SCHEMA:
            raise ContractError("decision schema is incompatible", code=FailureCode.INVALID_VALUE)
        workload = None if data["workload"] is None else FrozenWorkload.from_dict(data["workload"])
        candidate = None
        if data["candidate"] is not None:
            if workload is None:
                raise ContractError("decision candidate requires workload", code=FailureCode.INVALID_VALUE)
            candidate = CandidateProposal.from_dict(data["candidate"], workload)
        policy = None if data["policy"] is None else EvaluationPolicy.from_dict(data["policy"])
        runtime = None if data["runtime"] is None else RuntimeIdentity.from_dict(data["runtime"])
        if type(data["artifacts"]) is not list:
            raise ContractError("decision.artifacts must be an array", code=FailureCode.WRONG_TYPE)
        artifacts = tuple(Artifact.from_dict(item) for item in data["artifacts"])
        decision = cls(
            source=data["source"],
            action=data["action"],
            reason=data["reason"],
            receipt_id=data["receipt_id"],
            workload=workload,
            candidate=candidate,
            policy=policy,
            runtime=runtime,
            artifacts=artifacts,
            created_at_ns=data["created_at_ns"],
            claims=data["claims"],
            attestation=data["attestation"],
            decision_id=data["decision_id"],
        )
        if decision.decision_id != sha256_hex(decision._body_dict()):
            raise ContractError("decision_id does not match canonical decision body", code=FailureCode.IDENTITY_MISMATCH)
        return decision

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "PromotionDecision":
        return cls.from_dict(strict_json_loads(value))


Decision = PromotionDecision


def _fallback_from_validation(validation: ReceiptValidation, *, reason: str, now_ns: int | None) -> PromotionDecision:
    receipt = validation.receipt
    return PromotionDecision(
        source="promotion_rejected",
        action=NATIVE,
        reason=reason,
        receipt_id=receipt.receipt_id if receipt is not None else None,
        workload=receipt.workload if receipt is not None else None,
        candidate=receipt.candidate if receipt is not None else None,
        policy=receipt.policy if receipt is not None else None,
        runtime=receipt.runtime if receipt is not None else None,
        artifacts=receipt.artifacts if receipt is not None else (),
        attestation=None,
        created_at_ns=now_ns,
    )


def _independently_attested(validation: ReceiptValidation, key: bytes | bytearray | None) -> bool:
    """Verify the supervisor MAC without consulting ReceiptValidation._proof."""

    receipt = validation.receipt
    if receipt is None or type(validation.attestation) is not str:
        return False
    if type(key) not in {bytes, bytearray} or not bytes(key):
        return False
    try:
        expected = receipt_attestation(receipt, bytes(key))
    except ContractError:
        return False
    return hmac.compare_digest(validation.attestation, expected)


def make_promotion_decision(
    validation: ReceiptValidation,
    *,
    now_ns: int | None = None,
    attestation_key: bytes | bytearray | None = None,
) -> PromotionDecision:
    """Purely decide activation from validator output; never from raw evidence."""

    if not isinstance(validation, ReceiptValidation):
        raise ContractError(
            "promotion requires ReceiptValidation, not an observation or recommendation",
            code=FailureCode.PROMOTION_REJECTED,
        )
    independently_validated = False
    if validation.receipt is not None and _independently_attested(validation, attestation_key):
        checked = validate_receipt(
            validation.receipt,
            attestation=validation.attestation,
            attestation_key=attestation_key,
        )
        independently_validated = checked.valid and checked.complete and checked.local and checked.attested
    if (
        validation.receipt is None
        or not independently_validated
        or not validation.ok
        or not validation.valid
        or not validation.complete
        or not validation.local
        or not validation.artifacts_verified
    ):
        reason = validation.failures[0].code.value if validation.failures else "supervisor_attestation_required"
        return _fallback_from_validation(validation, reason=reason, now_ns=now_ns)
    # A signed receipt is admissible evidence even when the candidate is
    # slower or the result is inconclusive.  Activation reads the
    # independently recomputed Wave B statistics verdict -- never the bare
    # gain sign, and never caller-supplied metrics.  Missing/unparseable
    # statistical fields fail closed to not-promotable (never silently
    # treated as a pass).
    # checked.recomputed is frozen (see ReceiptValidation.__post_init__'s
    # _freeze_json call), so nested objects are MappingProxyType, not dict
    # -- isinstance(..., Mapping) is required here, not isinstance(..., dict).
    statistics = checked.recomputed.get("statistics")
    if not isinstance(statistics, Mapping):
        return _fallback_from_validation(validation, reason="statistics_missing", now_ns=now_ns)
    if statistics.get("calibration") is True:
        # A/A calibration receipts are valid evidence of the measured noise
        # floor, never a promotable candidate result -- see
        # auto_mlx.statistics's calibration field and "auto-mlx evaluate
        # --calibrate".
        return _fallback_from_validation(validation, reason="calibration_receipt_not_promotable", now_ns=now_ns)
    verdict = statistics.get("verdict")
    if verdict == "regressed":
        return _fallback_from_validation(validation, reason="regressed", now_ns=now_ns)
    if verdict == "inconclusive":
        return _fallback_from_validation(validation, reason="inconclusive", now_ns=now_ns)
    if verdict != "improved" or not (
        type(statistics.get("ci_lower_ns")) is int
        and type(statistics.get("min_effect_ns")) is int
        and statistics["ci_lower_ns"] > statistics["min_effect_ns"]
    ):
        return _fallback_from_validation(validation, reason="statistics_missing", now_ns=now_ns)
    receipt = validation.receipt
    return PromotionDecision(
        source="validated_local_receipt",
        action=ACTIVATE,
        reason="complete_validated_local_receipt",
        receipt_id=receipt.receipt_id,
        workload=receipt.workload,
        candidate=receipt.candidate,
        policy=receipt.policy,
        runtime=receipt.runtime,
        artifacts=receipt.artifacts,
        attestation=validation.attestation,
        created_at_ns=now_ns,
    )


def decision_identity(value: PromotionDecision | Mapping[str, Any]) -> str:
    decision = value if isinstance(value, PromotionDecision) else PromotionDecision.from_dict(value)
    return sha256_hex(decision._body_dict())


def write_decision(store: ContentAddressedStore, decision: PromotionDecision | Mapping[str, Any]) -> os.PathLike[str]:
    return store.put_decision(decision)


evaluate_promotion = make_promotion_decision
promote = make_promotion_decision
promote_receipt = make_promotion_decision


def activate(
    store: ContentAddressedStore,
    validation: ReceiptValidation,
    *,
    now_ns: int | None = None,
    artifact_root: str | os.PathLike[str] | None = None,
    attestation_key: bytes | bytearray | None = None,
) -> PromotionDecision:
    """Persist a pure decision and atomically publish its current pointer."""

    decision = make_promotion_decision(validation, now_ns=now_ns, attestation_key=attestation_key)
    if decision.action == ACTIVATE:
        try:
            if artifact_root is None or validation.receipt is None or decision.receipt_id is None:
                raise ContractError("activation requires an artifact root and stored receipt", code=FailureCode.PROMOTION_REJECTED)
            stored_receipt = store.get_receipt(decision.receipt_id)
            if stored_receipt.to_dict() != validation.receipt.to_dict():
                raise ContractError("stored receipt does not exactly match activation evidence", code=FailureCode.IDENTITY_MISMATCH)
            checked = validate_receipt(
                stored_receipt,
                workload=decision.workload,
                candidate=decision.candidate,
                policy=decision.policy,
                runtime=decision.runtime,
                artifact_root=artifact_root,
                attestation=decision.attestation,
                attestation_key=attestation_key,
            )
            if not (checked.valid and checked.complete and checked.local and checked.attested and checked.artifacts_verified):
                raise ContractError("stored receipt failed activation validation", code=FailureCode.PROMOTION_REJECTED)
        except ContractError as exc:
            decision = _fallback_from_validation(
                validation,
                reason=f"activation_rejected:{exc.code.value}",
                now_ns=now_ns,
            )
    try:
        store.put_decision(decision, require_durable=True)
        if decision.action == ACTIVATE:
            store.set_current_decision(decision.decision_id, require_durable=True)
        else:
            store.set_current_decision(NATIVE_FALLBACK, require_durable=True)
        return decision
    except BaseException as activation_error:
        # A rejected or interrupted activation must not leave an older
        # candidate selected.  The fallback write is attempted even when the
        # preceding durable operation failed; its failure is intentionally
        # surfaced to the caller.
        try:
            store.set_current_decision(NATIVE_FALLBACK, require_durable=False)
        except BaseException:
            raise activation_error
        raise activation_error


def rollback(
    store: ContentAddressedStore,
    *,
    reason: str = "operator_rollback",
    now_ns: int | None = None,
) -> PromotionDecision:
    """Write an immutable rollback decision and point dispatch at native code."""

    decision = PromotionDecision(
        source="rollback",
        action=NATIVE,
        reason=reason,
        receipt_id=None,
        workload=None,
        candidate=None,
        policy=None,
        runtime=None,
        artifacts=(),
        created_at_ns=now_ns,
    )
    try:
        store.put_decision(decision, require_durable=True)
        store.set_current_decision(NATIVE_FALLBACK, require_durable=True)
        return decision
    except BaseException as rollback_error:
        try:
            store.set_current_decision(NATIVE_FALLBACK, require_durable=False)
        except BaseException:
            raise rollback_error
        raise rollback_error


__all__: Final = [
    "ACTIVATE",
    "DECISION_SCHEMA",
    "Decision",
    "NATIVE",
    "PromotionDecision",
    "activate",
    "decision_identity",
    "evaluate_promotion",
    "make_promotion_decision",
    "promote",
    "promote_receipt",
    "rollback",
    "write_decision",
]
