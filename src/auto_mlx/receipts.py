"""Immutable local evidence receipts and their content-addressed store.

This module deliberately has no MLX dependency.  It is the trust boundary
between a local evaluator and the activation lane: all derived values are
recomputed from raw samples before a receipt can be considered valid.
"""

from __future__ import annotations

import errno
import base64
import hashlib
import hmac
import os
import stat
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol

from .canonical import canonical_bytes, canonical_json, sha256_hex, strict_json_loads, validate_json_value
from .contracts import (
    Artifact,
    CandidateProposal,
    EvaluationPolicy,
    FrozenWorkload,
    RuntimeIdentity,
)
from .errors import ContractError, Failure, FailureCode, UnknownFieldError
from .paths import _open_root_directory, validate_sha256, verify_artifact
from .statistics import VERDICT_INCONCLUSIVE, StatisticsVerdict, compute_sample_timing, compute_statistics_verdict


# v2 (Wave A measurement-integrity remediation): ExecutionRecord.record
# wire gained a required runner_elapsed_ns (the evidentiary runner-subprocess
# span, excluding authority-verification probe time); the evaluator_bundle
# wire gained thermal_blocks (per-block pmset preflight annotations) and a
# per-sample warm_state note.
#
# v3 (Wave B statistical-decision remediation): the evaluator_bundle wire's
# measurement_runs-sized block count became a variable, sequentially-
# extended block_count_used (policy.measurement_runs is now the starting/
# minimum count, policy.max_measurement_runs the cap); every warmup and
# measurement sample gained a sample_timing note (in-runner K-repetition
# min-of-K point estimate, or a forged_timing-degraded fallback -- see
# auto_mlx.statistics); and the receipt gained a top-level, nullable
# statistics field carrying the BCa-bootstrap verdict, CI bounds, seed, and
# threshold.  Older receipts do not carry these fields and are rejected
# outright rather than silently reinterpreted -- this store is
# local/experimental with no external consumers yet, and a strictly
# validated evidence format is safer without a dual-schema code path.
RECEIPT_SCHEMA: Final = "auto_mlx.receipt.v3"
NATIVE_FALLBACK: Final = "native_fallback"
CLAIMS_WITHHELD: Final = "withheld_pending_external_attestation"
_REQUIRED_ISOLATION: Final = frozenset({"network_denial", "descendant_containment"})
# These are deliberately conservative wire limits.  Reads use cap+1 so an
# oversized object is rejected before strict JSON parsing can accumulate it.
MAX_STORED_RECEIPT_BYTES: Final = 8 * 1024 * 1024
MAX_STORED_DECISION_BYTES: Final = 1 * 1024 * 1024
MAX_STORED_TUNING_SUMMARY_BYTES: Final = 1 * 1024 * 1024
MAX_STORED_HISTORY_INDEX_BYTES: Final = 1 * 1024 * 1024
MAX_CURRENT_POINTER_BYTES: Final = 256
_VALIDATION_TOKEN = object()
# Unlike aggregates/oracle/compatibility/metrics (never legitimately null),
# receipt.statistics IS legitimately null (raw-sample-lane receipts, or an
# evaluator_bundle whose measurements were never accepted) -- so "the
# caller did not pass statistics" needs a sentinel distinct from the real
# value None, or Receipt.from_dict reconstructing a stored null would be
# silently overwritten by a freshly recomputed non-null value instead of
# preserving (and later flagging, precisely) the stored one.
_MISSING_STATISTICS = object()


@dataclass(frozen=True, slots=True)
class _ValidationProof:
    token: object
    receipt_id: str | None
    attestation: str | None


def _attestation_key(value: bytes | bytearray | None) -> bytes:
    if type(value) not in {bytes, bytearray} or not value:
        raise ContractError(
            "an out-of-band supervisor attestation key is required",
            code=FailureCode.PROMOTION_REJECTED,
        )
    return bytes(value)


class ContractMapping(Protocol):
    """Narrow adapter accepted by the evidence lane."""

    def to_dict(self) -> dict[str, Any]: ...


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


def _string(value: Any, *, label: str, non_empty: bool = True) -> str:
    if type(value) is not str or (non_empty and not value):
        raise ContractError(f"{label} must be a {'non-empty ' if non_empty else ''}string", code=FailureCode.WRONG_TYPE)
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ContractError(f"{label} must not contain unpaired surrogates", code=FailureCode.INVALID_UNICODE)
    return value


def _integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ContractError(f"{label} must be an integer{suffix}", code=FailureCode.WRONG_TYPE)
    return value


def _boolean(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label} must be a boolean", code=FailureCode.WRONG_TYPE)
    return value


def _json_value(value: Any, *, label: str) -> Any:
    """Validate a JSON value using the repository's strict canonical rules."""

    try:
        canonical_bytes(value)
    except ContractError as exc:
        raise ContractError(f"{label} is not a canonical JSON value", code=exc.code) from exc
    return value


def _freeze_json(value: Any, *, path: str = "$") -> Any:
    value_type = type(value)
    if value is None or value_type is str or value_type is int or value_type is bool:
        return value
    if value_type is float:
        raise ContractError(f"floating-point value at {path} is not allowed", code=FailureCode.FLOAT_NOT_ALLOWED)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ContractError(f"object key at {path} must be a string", code=FailureCode.WRONG_TYPE)
            if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                raise ContractError(
                    f"JSON contains an unpaired UTF-16 surrogate in an object key at {path}",
                    code=FailureCode.INVALID_UNICODE,
                )
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise ContractError(f"unsupported JSON value at {path}: {type(value).__name__}", code=FailureCode.WRONG_TYPE)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _coerce_workload(value: FrozenWorkload | Mapping[str, Any]) -> FrozenWorkload:
    if isinstance(value, FrozenWorkload):
        return value
    return FrozenWorkload.from_dict(value)


def _coerce_policy(value: EvaluationPolicy | Mapping[str, Any]) -> EvaluationPolicy:
    if isinstance(value, EvaluationPolicy):
        return value
    return EvaluationPolicy.from_dict(value)


def _coerce_runtime(value: RuntimeIdentity | Mapping[str, Any]) -> RuntimeIdentity:
    if isinstance(value, RuntimeIdentity):
        return value
    return RuntimeIdentity.from_dict(value)


def _coerce_failure(value: Failure | Mapping[str, Any] | None) -> Failure | None:
    if value is None or isinstance(value, Failure):
        return value
    data = _object(value, label="failure")
    _exact(data, {"code", "message", "details"}, label="failure")
    try:
        code = FailureCode(data["code"])
    except (TypeError, ValueError) as exc:
        raise ContractError("failure.code is not a known failure code", code=FailureCode.INVALID_VALUE) from exc
    details = _object(data["details"], label="failure.details")
    _json_value(details, label="failure.details")
    return Failure(code, _string(data["message"], label="failure.message"), details)


def _coerce_candidate(
    value: CandidateProposal | Mapping[str, Any], workload: FrozenWorkload
) -> CandidateProposal:
    if isinstance(value, CandidateProposal):
        if value.workload_hash != workload.workload_hash:
            raise ContractError("candidate does not belong to workload", code=FailureCode.IDENTITY_MISMATCH)
        return value
    return CandidateProposal.from_dict(value, workload)


def receipt_attestation(
    receipt: "Receipt", key: bytes | bytearray
) -> str:
    """Return the supervisor/evaluator HMAC for a receipt body.

    The key is intentionally accepted only as an out-of-band argument and is
    never serialized into a receipt, decision, or dispatch result.
    """

    if not isinstance(receipt, Receipt):
        raise ContractError("attestation requires a Receipt", code=FailureCode.WRONG_TYPE)
    secret = _attestation_key(key)
    return hmac.new(secret, canonical_bytes(receipt._body_dict()), hashlib.sha256).hexdigest()


attest_receipt = receipt_attestation


@dataclass(frozen=True, slots=True)
class RawSample:
    """One unaggregated measurement and its source-oracle comparison."""

    index: int
    duration_ns: int
    baseline_duration_ns: int
    actual_output: Any
    oracle_output: Any
    drift: int

    def __post_init__(self) -> None:
        _integer(self.index, label="raw_sample.index", minimum=0)
        _integer(self.duration_ns, label="raw_sample.duration_ns", minimum=0)
        _integer(self.baseline_duration_ns, label="raw_sample.baseline_duration_ns", minimum=0)
        _json_value(self.actual_output, label="raw_sample.actual_output")
        _json_value(self.oracle_output, label="raw_sample.oracle_output")
        _integer(self.drift, label="raw_sample.drift", minimum=0)
        object.__setattr__(self, "actual_output", _freeze_json(self.actual_output, path="raw_sample.actual_output"))
        object.__setattr__(self, "oracle_output", _freeze_json(self.oracle_output, path="raw_sample.oracle_output"))

    @property
    def sample_index(self) -> int:
        return self.index

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "duration_ns": self.duration_ns,
            "baseline_duration_ns": self.baseline_duration_ns,
            "actual_output": _thaw_json(self.actual_output),
            "oracle_output": _thaw_json(self.oracle_output),
            "drift": self.drift,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RawSample":
        data = _object(value, label="raw sample")
        _exact(
            data,
            {"index", "duration_ns", "baseline_duration_ns", "actual_output", "oracle_output", "drift"},
            label="raw sample",
        )
        return cls(
            data["index"],
            data["duration_ns"],
            data["baseline_duration_ns"],
            data["actual_output"],
            data["oracle_output"],
            data["drift"],
        )


def _stats(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {
            "count": 0,
            "sum_ns": 0,
            "min_ns": 0,
            "max_ns": 0,
            "mean_numerator": 0,
            "mean_denominator": 1,
            "range_ns": 0,
        }
    total = sum(values)
    minimum = min(values)
    maximum = max(values)
    return {
        "count": len(values),
        "sum_ns": total,
        "min_ns": minimum,
        "max_ns": maximum,
        "mean_numerator": total,
        "mean_denominator": len(values),
        "range_ns": maximum - minimum,
    }


def _mean_abs_deviation(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"numerator": 0, "denominator": 1}
    total = sum(values)
    return {
        "numerator": sum(abs(len(values) * value - total) for value in values),
        "denominator": len(values) * len(values),
    }


def recompute_receipt_fields(
    raw_samples: Sequence[RawSample],
    policy: EvaluationPolicy,
    *,
    evaluator_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute every derived receipt field from raw samples only."""

    if evaluator_bundle is not None:
        return _recompute_evaluator_bundle_fields(evaluator_bundle, policy)

    candidate_durations = [sample.duration_ns for sample in raw_samples]
    baseline_durations = [sample.baseline_duration_ns for sample in raw_samples]
    candidate = _stats(candidate_durations)
    baseline = _stats(baseline_durations)
    outcomes = []
    drift_values = [sample.drift for sample in raw_samples]
    for sample in raw_samples:
        actual_digest = sha256_hex(sample.actual_output)
        oracle_digest = sha256_hex(sample.oracle_output)
        outcomes.append(
            {
                "index": sample.index,
                "actual_sha256": actual_digest,
                "oracle_sha256": oracle_digest,
                "match": actual_digest == oracle_digest,
            }
        )
    matches = sum(1 for outcome in outcomes if outcome["match"])
    observed = len(raw_samples)
    compatibility = {
        "warmup_runs": policy.warmup_runs,
        "measurement_runs": policy.measurement_runs,
        "observed_samples": observed,
        "compatible": observed == policy.measurement_runs
        and [sample.index for sample in raw_samples] == list(range(policy.measurement_runs)),
        # The raw-sample lane predates thermal gating and carries no
        # per-block preflight data of its own; the evaluator-bundle lane
        # (the production path -- see _validate_evaluator_bundle_wire)
        # populates these from real pmset preflights.
        "thermal_blocks": [],
        "thermal_gate_policy": None,
        # The raw-sample lane predates Wave B sequential sampling too: it
        # has no paired-block structure to extend, so its block count is
        # always exactly policy.measurement_runs.
        "block_count_used": policy.measurement_runs,
    }
    gain_numerator = baseline["sum_ns"] - candidate["sum_ns"]
    return {
        "aggregates": {"candidate": candidate, "baseline": baseline},
        "oracle": {
            "sample_count": observed,
            "matches": matches,
            "mismatches": observed - matches,
            "all_match": observed == matches,
            "outcomes": outcomes,
        },
        "compatibility": compatibility,
        # The raw-sample lane has no per-block/per-iteration evidence to
        # derive a statistical verdict from -- see the evaluator_bundle
        # lane (_validate_evaluator_bundle_wire) for the real computation.
        "statistics": None,
        "metrics": {
            "drift": {
                "count": len(drift_values),
                "sum": sum(drift_values),
                "min": min(drift_values) if drift_values else 0,
                "max": max(drift_values) if drift_values else 0,
            },
            "dispersion": {
                "candidate": _mean_abs_deviation(candidate_durations),
                "baseline": _mean_abs_deviation(baseline_durations),
                "candidate_range_ns": candidate["range_ns"],
                "baseline_range_ns": baseline["range_ns"],
            },
            "gain": {
                "baseline_sum_ns": baseline["sum_ns"],
                "candidate_sum_ns": candidate["sum_ns"],
                "delta_ns": gain_numerator,
                "improved": gain_numerator > 0,
                "numerator": gain_numerator,
                "denominator": baseline["sum_ns"] if baseline["sum_ns"] else 1,
            },
        },
    }


def _validate_stats(value: Any, *, label: str) -> None:
    data = _object(value, label=label)
    _exact(
        data,
        {"count", "sum_ns", "min_ns", "max_ns", "mean_numerator", "mean_denominator", "range_ns"},
        label=label,
    )
    for field in data:
        _integer(data[field], label=f"{label}.{field}", minimum=0)


_THERMAL_STATES: Final = frozenset({"nominal", "throttled", "unknown"})


def _validate_thermal_reading_shape(value: Any, *, label: str) -> None:
    data = _object(value, label=label)
    _exact(
        data,
        {"state", "cpu_speed_limit_percent", "cpu_scheduler_limit_percent", "thermal_pressure_level", "detail"},
        label=label,
    )
    if data["state"] not in _THERMAL_STATES:
        raise ContractError(f"{label}.state is not a closed thermal state", code=FailureCode.INVALID_VALUE)
    for field in ("cpu_speed_limit_percent", "cpu_scheduler_limit_percent"):
        if data[field] is not None:
            _integer(data[field], label=f"{label}.{field}", minimum=0)
    if data["thermal_pressure_level"] is not None:
        _string(data["thermal_pressure_level"], label=f"{label}.thermal_pressure_level", non_empty=False)
    _string(data["detail"], label=f"{label}.detail", non_empty=False)


def _validate_thermal_block_shape(value: Any, *, label: str) -> dict[str, Any]:
    """Structural-only validation: no policy/block-identity binding here.

    Thermal readings have no raw evidence to recompute them from (they are
    a point-in-time OS reading, not something a runner subprocess
    produces), so this validator's job is to *tolerate and preserve* a
    well-formed annotation, not to re-derive its numbers.  Identity/policy
    binding is layered on top by ``_validate_evaluator_bundle_wire``, which
    has the policy and block context this shape check does not.
    """

    data = _object(value, label=label)
    _exact(data, {"block_id", "block_index", "policy", "preflight", "refused"}, label=label)
    _string(data["block_id"], label=f"{label}.block_id")
    _integer(data["block_index"], label=f"{label}.block_index", minimum=0)
    if data["policy"] not in {"tag", "refuse"}:
        raise ContractError(f"{label}.policy is not a closed thermal gate policy", code=FailureCode.INVALID_VALUE)
    _boolean(data["refused"], label=f"{label}.refused")
    preflight = _object(data["preflight"], label=f"{label}.preflight")
    _exact(preflight, {"initial", "final", "retried", "thermally_suspect"}, label=f"{label}.preflight")
    _boolean(preflight["retried"], label=f"{label}.preflight.retried")
    _boolean(preflight["thermally_suspect"], label=f"{label}.preflight.thermally_suspect")
    _validate_thermal_reading_shape(preflight["initial"], label=f"{label}.preflight.initial")
    _validate_thermal_reading_shape(preflight["final"], label=f"{label}.preflight.final")
    if data["refused"] and (data["policy"] != "refuse" or not preflight["thermally_suspect"]):
        raise ContractError(
            f"{label} cannot be refused without a matching suspect reading and refuse policy",
            code=FailureCode.INVALID_VALUE,
        )
    return data


def _validate_derived_fields(
    aggregates: Any,
    oracle: Any,
    compatibility: Any,
    metrics: Any,
    statistics: Any = None,
) -> None:
    aggregate_data = _object(aggregates, label="receipt.aggregates")
    _exact(aggregate_data, {"candidate", "baseline"}, label="receipt.aggregates")
    _validate_stats(aggregate_data["candidate"], label="receipt.aggregates.candidate")
    _validate_stats(aggregate_data["baseline"], label="receipt.aggregates.baseline")

    oracle_data = _object(oracle, label="receipt.oracle")
    _exact(oracle_data, {"sample_count", "matches", "mismatches", "all_match", "outcomes"}, label="receipt.oracle")
    for field in ("sample_count", "matches", "mismatches"):
        _integer(oracle_data[field], label=f"receipt.oracle.{field}", minimum=0)
    _boolean(oracle_data["all_match"], label="receipt.oracle.all_match")
    if type(oracle_data["outcomes"]) is not list:
        raise ContractError("receipt.oracle.outcomes must be an array", code=FailureCode.WRONG_TYPE)
    for index, outcome in enumerate(oracle_data["outcomes"]):
        outcome_data = _object(outcome, label=f"receipt.oracle.outcomes[{index}]")
        _exact(
            outcome_data,
            {"index", "actual_sha256", "oracle_sha256", "match"},
            label=f"receipt.oracle.outcomes[{index}]",
        )
        _integer(outcome_data["index"], label=f"receipt.oracle.outcomes[{index}].index", minimum=0)
        validate_sha256(outcome_data["actual_sha256"])
        validate_sha256(outcome_data["oracle_sha256"])
        _boolean(outcome_data["match"], label=f"receipt.oracle.outcomes[{index}].match")

    compatibility_data = _object(compatibility, label="receipt.compatibility")
    _exact(
        compatibility_data,
        {
            "warmup_runs", "measurement_runs", "observed_samples", "compatible", "thermal_blocks",
            "thermal_gate_policy", "block_count_used",
        },
        label="receipt.compatibility",
    )
    for field in ("warmup_runs", "measurement_runs", "observed_samples"):
        _integer(compatibility_data[field], label=f"receipt.compatibility.{field}", minimum=0)
    _integer(compatibility_data["block_count_used"], label="receipt.compatibility.block_count_used", minimum=1)
    _boolean(compatibility_data["compatible"], label="receipt.compatibility.compatible")
    if compatibility_data["thermal_gate_policy"] is not None and compatibility_data["thermal_gate_policy"] not in {"tag", "refuse"}:
        raise ContractError("receipt.compatibility.thermal_gate_policy is not a closed thermal gate policy", code=FailureCode.INVALID_VALUE)
    if type(compatibility_data["thermal_blocks"]) is not list:
        raise ContractError("receipt.compatibility.thermal_blocks must be an array", code=FailureCode.WRONG_TYPE)
    for index, entry in enumerate(compatibility_data["thermal_blocks"]):
        _validate_thermal_block_shape(entry, label=f"receipt.compatibility.thermal_blocks[{index}]")

    metrics_data = _object(metrics, label="receipt.metrics")
    _exact(metrics_data, {"drift", "dispersion", "gain"}, label="receipt.metrics")
    drift = _object(metrics_data["drift"], label="receipt.metrics.drift")
    _exact(drift, {"count", "sum", "min", "max"}, label="receipt.metrics.drift")
    for field in drift:
        _integer(drift[field], label=f"receipt.metrics.drift.{field}", minimum=0)
    dispersion = _object(metrics_data["dispersion"], label="receipt.metrics.dispersion")
    _exact(
        dispersion,
        {"candidate", "baseline", "candidate_range_ns", "baseline_range_ns"},
        label="receipt.metrics.dispersion",
    )
    _validate_stats_like_deviation(dispersion["candidate"], label="receipt.metrics.dispersion.candidate")
    _validate_stats_like_deviation(dispersion["baseline"], label="receipt.metrics.dispersion.baseline")
    for field in ("candidate_range_ns", "baseline_range_ns"):
        _integer(dispersion[field], label=f"receipt.metrics.dispersion.{field}", minimum=0)
    gain = _object(metrics_data["gain"], label="receipt.metrics.gain")
    _exact(
        gain,
        {"baseline_sum_ns", "candidate_sum_ns", "delta_ns", "improved", "numerator", "denominator"},
        label="receipt.metrics.gain",
    )
    for field in ("baseline_sum_ns", "candidate_sum_ns"):
        _integer(gain[field], label=f"receipt.metrics.gain.{field}", minimum=0)
    _integer(gain["numerator"], label="receipt.metrics.gain.numerator")
    _integer(gain["denominator"], label="receipt.metrics.gain.denominator", minimum=1)
    _integer(gain["delta_ns"], label="receipt.metrics.gain.delta_ns")
    _boolean(gain["improved"], label="receipt.metrics.gain.improved")

    if statistics is not None:
        try:
            StatisticsVerdict.from_dict(statistics)
        except ContractError as exc:
            raise ContractError(f"receipt.statistics is malformed: {exc}", code=exc.code) from exc


def _validate_stats_like_deviation(value: Any, *, label: str) -> None:
    data = _object(value, label=label)
    _exact(data, {"numerator", "denominator"}, label=label)
    _integer(data["numerator"], label=f"{label}.numerator", minimum=0)
    _integer(data["denominator"], label=f"{label}.denominator", minimum=1)


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: Any, *, label: str) -> bytes:
    if type(value) is not str:
        raise ContractError(f"{label} must be base64 text", code=FailureCode.WRONG_TYPE)
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ContractError(f"{label} is not valid base64", code=FailureCode.INVALID_VALUE) from exc


def _record_to_wire(record: Any) -> dict[str, Any]:
    data = dict(record.to_dict())
    data["stdout_b64"] = _encode_bytes(record.stdout)
    data["stderr_b64"] = _encode_bytes(record.stderr)
    return data


def _oracle_to_wire(oracle: Any | None) -> dict[str, Any] | None:
    return None if oracle is None else dict(oracle.to_dict())


# Must stay byte-identical to auto_mlx.runners.reference_matmul.WARMUP_MARKER
# (that module cannot import auto_mlx -- see its own docstring).
_WARMUP_MARKER: Final = b"auto_mlx_runner_warmup_complete"


def _warm_state_for(stderr: bytes | None, *, arm: str, seen_arms: set) -> dict[str, Any]:
    """Compute a diagnostic warm/cold note for one sample -- never gates acceptance.

    Both fields are pure functions of already-evidentiary data: the marker
    check reads the record's ``stderr`` bytes (immutable, hash-bound bytes
    already part of the record on both the live ``ExecutionRecord`` and the
    wire) and ``first_launch_of_config_in_evaluate`` is derived from
    ``seen_arms``, the set of arms already observed earlier in this same
    evaluator bundle's canonical order (warmups, then measurement blocks in
    block/slot order).  Calling this identically at wire-build time (with
    ``record.stderr``) and at validation time (with the decoded
    ``stderr_b64`` bytes) -- over the same ordered iteration -- is what
    makes ``warm_state`` independently recomputed, not merely stored.
    """

    first_launch = arm not in seen_arms
    seen_arms.add(arm)
    marker_present = stderr is not None and _WARMUP_MARKER in stderr
    return {
        "in_runner_warmup_marker_present": marker_present,
        "first_launch_of_config_in_evaluate": first_launch,
    }


def _validate_warm_state_shape(value: Any, *, label: str) -> dict[str, Any]:
    data = _object(value, label=label)
    _exact(data, {"in_runner_warmup_marker_present", "first_launch_of_config_in_evaluate"}, label=label)
    _boolean(data["in_runner_warmup_marker_present"], label=f"{label}.in_runner_warmup_marker_present")
    _boolean(data["first_launch_of_config_in_evaluate"], label=f"{label}.first_launch_of_config_in_evaluate")
    return data


def _sample_timing_for(*, runner_elapsed_ns: int | None, parent_elapsed_ns: int, stderr: bytes | None) -> dict[str, Any]:
    """Wave B per-sample timing note; tolerant like ``_warm_state_for`` above.

    Delegates the real trust-boundary work (forged-timing cross-check,
    min-of-K reduction) to ``auto_mlx.statistics.compute_sample_timing``.
    This wrapper only handles the cases that function deliberately refuses
    to guess at: no usable ``runner_elapsed_ns`` at all (a failed sample, or
    a missing measurement slot) degrades to a ``parent_elapsed_ns``-based
    placeholder -- never a crash, and never treated as trusted evidence.
    Calling this identically at wire-build time (from a live
    ``ExecutionRecord``) and at validation time (from the decoded wire) is
    what makes ``sample_timing`` independently recomputed, not merely
    stored -- exactly the ``warm_state`` pattern above.
    """

    if runner_elapsed_ns is None or runner_elapsed_ns <= 0:
        return {
            "raw_iterations_ns": [],
            "trusted": False,
            "point_estimate_ns": max(1, parent_elapsed_ns),
            "rejection_reason": None,
        }
    return compute_sample_timing(runner_elapsed_ns, stderr if stderr is not None else b"").to_dict()


def _sample_timing_for_record(record: Any | None) -> dict[str, Any]:
    if record is None:
        return {"raw_iterations_ns": [], "trusted": False, "point_estimate_ns": 1, "rejection_reason": None}
    return _sample_timing_for(runner_elapsed_ns=record.runner_elapsed_ns, parent_elapsed_ns=record.parent_elapsed_ns, stderr=record.stderr)


def _validate_sample_timing_shape(value: Any, *, label: str) -> dict[str, Any]:
    data = _object(value, label=label)
    _exact(data, {"raw_iterations_ns", "trusted", "point_estimate_ns", "rejection_reason"}, label=label)
    if type(data["raw_iterations_ns"]) is not list or any(
        type(item) is not int or item <= 0 for item in data["raw_iterations_ns"]
    ):
        raise ContractError(f"{label}.raw_iterations_ns must be an array of positive integers", code=FailureCode.WRONG_TYPE)
    _boolean(data["trusted"], label=f"{label}.trusted")
    _integer(data["point_estimate_ns"], label=f"{label}.point_estimate_ns", minimum=1)
    if data["rejection_reason"] is not None and data["rejection_reason"] != "forged_timing":
        raise ContractError(f"{label}.rejection_reason must be null or forged_timing", code=FailureCode.INVALID_VALUE)
    if data["trusted"] and data["rejection_reason"] is not None:
        raise ContractError(f"{label} cannot be trusted and carry a rejection_reason", code=FailureCode.INVALID_VALUE)
    if data["trusted"] and not data["raw_iterations_ns"]:
        raise ContractError(f"{label} cannot be trusted without raw iterations", code=FailureCode.INVALID_VALUE)
    return data


def _measurement_bundle_to_wire(measurements: Any, *, seen_arms: set) -> dict[str, Any]:
    blocks = []
    for observation in measurements.blocks:
        blocks.append(
            {
                "block_id": observation.block.block_id,
                "block_index": observation.block.block_index,
                "sequence": list(observation.block.sequence),
                "accepted": observation.accepted,
                "rejection_reasons": list(observation.rejection_reasons),
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "block_id": sample.block_id,
                        "slot_index": sample.slot_index,
                        "arm": sample.arm,
                        "record": None if sample.record is None else _record_to_wire(sample.record),
                        "oracle": _oracle_to_wire(sample.oracle),
                        "warm_state": _warm_state_for(
                            sample.record.stderr if sample.record else None, arm=sample.arm, seen_arms=seen_arms
                        ),
                        "sample_timing": _sample_timing_for_record(sample.record),
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
        )
    return {
        "plan_digest": measurements.plan_digest,
        "accepted": measurements.accepted,
        "rejection_reasons": list(measurements.rejection_reasons),
        "unexpected_sample_ids": list(measurements.unexpected_sample_ids),
        "blocks": blocks,
    }


def _observation_bundle_to_wire(
    bundle: Any,
    policy: EvaluationPolicy,
    oracle: Any | None,
    *,
    measurements: Any | None = None,
) -> dict[str, Any]:
    from .oracle import ExactOutputOracle, OracleDescriptor

    if oracle is None:
        raise ContractError(
            "an evaluator-owned exact oracle is required to adapt an observation bundle",
            code=FailureCode.ORACLE_MISMATCH,
        )
    if not isinstance(oracle, ExactOutputOracle):
        raise ContractError("observation bundle oracle must be an ExactOutputOracle", code=FailureCode.ORACLE_MISMATCH)
    if getattr(bundle, "oracle", None) is not oracle:
        raise ContractError(
            "observation bundle must retain the evaluator constructor's original oracle",
            code=FailureCode.IDENTITY_MISMATCH,
        )
    oracle_descriptor = getattr(bundle, "oracle_descriptor", None)
    if type(oracle_descriptor) is not OracleDescriptor or oracle_descriptor != oracle.descriptor:
        raise ContractError(
            "observation bundle oracle is not bound to the evaluator oracle descriptor",
            code=FailureCode.IDENTITY_MISMATCH,
        )
    measurements = bundle.measurements if measurements is None else measurements
    if type(bundle.isolation_provider_id) not in {str, type(None)}:
        raise ContractError("observation isolation provider identity is malformed", code=FailureCode.WRONG_TYPE)
    if type(bundle.isolation_identity) not in {str, type(None)}:
        raise ContractError("observation isolation identity is malformed", code=FailureCode.WRONG_TYPE)
    isolation_verifier_id = getattr(bundle, "isolation_verifier_id", None)
    isolation_verifier_identity = getattr(bundle, "isolation_verifier_identity", None)
    if type(isolation_verifier_id) not in {str, type(None)}:
        raise ContractError("observation isolation verifier identity is malformed", code=FailureCode.WRONG_TYPE)
    if type(isolation_verifier_identity) not in {str, type(None)}:
        raise ContractError("observation isolation verifier digest is malformed", code=FailureCode.WRONG_TYPE)
    isolation_requirements = getattr(bundle, "isolation_requirements", None)
    if isolation_requirements is None and bundle.isolation_provider_id is not None:
        isolation_requirements = _REQUIRED_ISOLATION
    if isolation_requirements is not None:
        if type(isolation_requirements) not in {set, frozenset} or any(
            type(item) is not str or not item for item in isolation_requirements
        ):
            raise ContractError("observation isolation requirements are malformed", code=FailureCode.WRONG_TYPE)
        isolation_requirements = frozenset(isolation_requirements)
    isolation_values = (
        bundle.isolation_provider_id,
        bundle.isolation_identity,
        isolation_verifier_id,
        isolation_verifier_identity,
        isolation_requirements,
    )
    if any(value is not None for value in isolation_values) and not all(value is not None for value in isolation_values):
        raise ContractError("observation isolation provenance must be complete or absent", code=FailureCode.INVALID_VALUE)
    if isolation_requirements is not None and not _REQUIRED_ISOLATION.issubset(isolation_requirements):
        raise ContractError("observation isolation requirements are incomplete", code=FailureCode.SANDBOX_UNAVAILABLE)
    if bundle.isolation_identity is not None:
        validate_sha256(bundle.isolation_identity)
    if isolation_verifier_identity is not None:
        validate_sha256(isolation_verifier_identity)
    thermal_blocks = getattr(bundle, "thermal_blocks", ())
    if type(thermal_blocks) not in {tuple, list} or any(type(item) is not dict for item in thermal_blocks):
        raise ContractError("observation bundle thermal_blocks must be an array of objects", code=FailureCode.WRONG_TYPE)
    statistics = getattr(bundle, "statistics", None)
    if statistics is not None and type(statistics) is not dict and not isinstance(statistics, Mapping):
        raise ContractError("observation bundle statistics must be an object or null", code=FailureCode.WRONG_TYPE)
    seen_arms: set = set()
    warmups_wire = [
        {
            "sample_id": observation.sample_id,
            "arm": observation.arm,
            "record": _record_to_wire(observation.record),
            "oracle": _oracle_to_wire(observation.oracle),
            "warm_state": _warm_state_for(observation.record.stderr, arm=observation.arm, seen_arms=seen_arms),
            "sample_timing": _sample_timing_for_record(observation.record),
        }
        for observation in bundle.warmups
    ]
    measurements_wire = _measurement_bundle_to_wire(measurements, seen_arms=seen_arms)
    return {
        "schema": "auto_mlx.observation_bundle.v3",
        "candidate_id": bundle.candidate_id,
        "workload_hash": bundle.workload_hash,
        "runtime": bundle.runtime.to_dict(),
        "policy": policy.to_dict(),
        "baseline_runner_id": bundle.baseline_runner_id,
        "baseline_runner_digest": bundle.baseline_runner_digest,
        "candidate_runner_id": bundle.candidate_runner_id,
        "candidate_runner_digest": bundle.candidate_runner_digest,
        "isolation_provider_id": bundle.isolation_provider_id,
        "isolation_identity": bundle.isolation_identity,
        "isolation_verifier_id": isolation_verifier_id,
        "isolation_verifier_identity": isolation_verifier_identity,
        "isolation_requirements": sorted(isolation_requirements) if isolation_requirements is not None else None,
        "oracle": {
            "label": oracle_descriptor.label,
            "expected_digest": oracle_descriptor.expected_digest,
            "expected_size": oracle_descriptor.expected_size,
            "expected_b64": _encode_bytes(oracle.expected),
        },
        "warmups": warmups_wire,
        "measurements": measurements_wire,
        "thermal_blocks": [dict(item) for item in thermal_blocks],
        "statistics": None if statistics is None else dict(statistics),
    }


def _validate_record_wire(value: Any, *, label: str) -> tuple[bytes, bytes, dict[str, Any]]:
    data = _object(value, label=label)
    expected = {
        "candidate_id", "workload_hash", "runner_id", "runner_digest", "status", "parent_elapsed_ns",
        "runner_elapsed_ns", "observation_id", "arm", "returncode", "stdout_sha256", "stderr_sha256",
        "stdout_bytes", "stderr_bytes", "stdout_truncated", "stderr_truncated", "output_truncated",
        "isolation", "cleanup", "failure", "stdout_b64", "stderr_b64",
    }
    _exact(data, expected, label=label)
    validate_sha256(data["runner_digest"])
    _string(data["candidate_id"], label=f"{label}.candidate_id")
    validate_sha256(data["workload_hash"])
    _string(data["runner_id"], label=f"{label}.runner_id")
    _string(data["status"], label=f"{label}.status")
    if data["status"] not in {"success", "exit_failure", "crash", "timeout", "output_failure", "sandbox_unavailable", "start_failure", "artifact_failure"}:
        raise ContractError(f"{label}.status is not a closed execution status", code=FailureCode.INVALID_VALUE)
    elapsed = _integer(data["parent_elapsed_ns"], label=f"{label}.parent_elapsed_ns", minimum=0)
    if data["runner_elapsed_ns"] is not None:
        _integer(data["runner_elapsed_ns"], label=f"{label}.runner_elapsed_ns", minimum=1)
    if data["status"] == "success" and data["runner_elapsed_ns"] is None:
        raise ContractError(f"{label}.success requires a runner_elapsed_ns", code=FailureCode.INVALID_VALUE)
    if data["observation_id"] is not None:
        _string(data["observation_id"], label=f"{label}.observation_id")
    if data["arm"] is not None and data["arm"] not in {"baseline", "candidate"}:
        raise ContractError(f"{label}.arm is invalid", code=FailureCode.INVALID_VALUE)
    if data["returncode"] is not None:
        if type(data["returncode"]) is not int:
            raise ContractError(f"{label}.returncode must be an integer or null", code=FailureCode.WRONG_TYPE)
    stdout = _decode_bytes(data["stdout_b64"], label=f"{label}.stdout_b64")
    stderr = _decode_bytes(data["stderr_b64"], label=f"{label}.stderr_b64")
    validate_sha256(data["stdout_sha256"])
    validate_sha256(data["stderr_sha256"])
    for name in ("stdout_bytes", "stderr_bytes"):
        _integer(data[name], label=f"{label}.{name}", minimum=0)
    if hashlib.sha256(stdout).hexdigest() != data["stdout_sha256"] or len(stdout) != data["stdout_bytes"]:
        raise ContractError(f"{label}.stdout bytes do not match its digest/size", code=FailureCode.IDENTITY_MISMATCH)
    if hashlib.sha256(stderr).hexdigest() != data["stderr_sha256"] or len(stderr) != data["stderr_bytes"]:
        raise ContractError(f"{label}.stderr bytes do not match its digest/size", code=FailureCode.IDENTITY_MISMATCH)
    for name in ("stdout_truncated", "stderr_truncated", "output_truncated"):
        _boolean(data[name], label=f"{label}.{name}")
    failure = _coerce_failure(data["failure"])
    status = data["status"]
    if status == "success":
        if data["returncode"] != 0:
            raise ContractError(f"{label}.success requires returncode 0", code=FailureCode.RUNTIME_FAILURE)
        if failure is not None:
            raise ContractError(f"{label}.success cannot carry a failure", code=FailureCode.RUNTIME_FAILURE)
        if data["stdout_truncated"] or data["stderr_truncated"] or data["output_truncated"]:
            raise ContractError(f"{label}.success cannot carry truncated output", code=FailureCode.OUTPUT_LIMIT)
    elif status == "exit_failure":
        if data["returncode"] is None or data["returncode"] == 0:
            raise ContractError(f"{label}.exit_failure requires a non-zero returncode", code=FailureCode.RUNTIME_FAILURE)
        if failure is None:
            raise ContractError(f"{label}.exit_failure requires a failure classification", code=FailureCode.RUNTIME_FAILURE)
    elif status == "crash":
        if data["returncode"] is not None and data["returncode"] >= 0:
            raise ContractError(f"{label}.crash requires a negative or null returncode", code=FailureCode.RUNTIME_FAILURE)
        if failure is None:
            raise ContractError(f"{label}.crash requires a failure classification", code=FailureCode.RUNTIME_FAILURE)
    elif failure is None:
        raise ContractError(f"{label}.{status} requires a failure classification", code=FailureCode.RUNTIME_FAILURE)
    isolation = data["isolation"]
    if isolation is not None:
        isolation_data = _object(isolation, label=f"{label}.isolation")
        _exact(
            isolation_data,
            {"provider_id", "identity", "verifier_id", "verifier_identity", "requirements", "attestation_digest", "production_eligible"},
            label=f"{label}.isolation",
        )
        _string(isolation_data["provider_id"], label=f"{label}.isolation.provider_id")
        validate_sha256(isolation_data["identity"])
        _string(isolation_data["verifier_id"], label=f"{label}.isolation.verifier_id")
        validate_sha256(isolation_data["verifier_identity"])
        validate_sha256(isolation_data["attestation_digest"])
        _boolean(isolation_data["production_eligible"], label=f"{label}.isolation.production_eligible")
        if isolation_data["production_eligible"]:
            raise ContractError(f"{label}.isolation cannot claim G0 production eligibility", code=FailureCode.SANDBOX_UNAVAILABLE)
        if type(isolation_data["requirements"]) is not list or any(type(item) is not str for item in isolation_data["requirements"]):
            raise ContractError(f"{label}.isolation.requirements must be a string array", code=FailureCode.WRONG_TYPE)
        if not {"network_denial", "descendant_containment"}.issubset(isolation_data["requirements"]):
            raise ContractError(f"{label}.isolation lacks required containment proof", code=FailureCode.SANDBOX_UNAVAILABLE)
    cleanup = _object(data["cleanup"], label=f"{label}.cleanup")
    _exact(cleanup, {"attempted", "mode", "verified"}, label=f"{label}.cleanup")
    _boolean(cleanup["attempted"], label=f"{label}.cleanup.attempted")
    _string(cleanup["mode"], label=f"{label}.cleanup.mode")
    _boolean(cleanup["verified"], label=f"{label}.cleanup.verified")
    if cleanup["verified"]:
        raise ContractError(f"{label}.cleanup cannot certify containment", code=FailureCode.SANDBOX_UNAVAILABLE)
    return stdout, stderr, data


def _validate_oracle_wire(value: Any, stdout: bytes, *, label: str) -> tuple[bool, str, str]:
    data = _object(value, label=label)
    _exact(data, {"matched", "expected_digest", "actual_digest", "expected_size", "actual_size", "failure"}, label=label)
    _boolean(data["matched"], label=f"{label}.matched")
    validate_sha256(data["expected_digest"])
    validate_sha256(data["actual_digest"])
    _integer(data["expected_size"], label=f"{label}.expected_size", minimum=0)
    _integer(data["actual_size"], label=f"{label}.actual_size", minimum=0)
    if data["failure"] is not None:
        _coerce_failure(data["failure"])
    actual_digest = hashlib.sha256(stdout).hexdigest()
    actual_size = len(stdout)
    if actual_digest != data["actual_digest"] or actual_size != data["actual_size"]:
        raise ContractError(f"{label} actual output is not independently verifiable", code=FailureCode.ORACLE_MISMATCH)
    expected_match = actual_digest == data["expected_digest"] and actual_size == data["expected_size"]
    if bool(data["matched"]) != expected_match:
        raise ContractError(f"{label}.matched is not derived from output bytes", code=FailureCode.ORACLE_MISMATCH)
    return expected_match, data["expected_digest"], actual_digest


def _validate_evaluator_bundle_wire(value: Mapping[str, Any], policy: EvaluationPolicy) -> dict[str, Any]:
    data = _object(value, label="receipt.evaluator_bundle")
    _exact(
        data,
        {
            "schema", "candidate_id", "workload_hash", "runtime", "policy", "baseline_runner_id",
            "baseline_runner_digest", "candidate_runner_id", "candidate_runner_digest",
            "isolation_provider_id", "isolation_identity", "isolation_verifier_id", "isolation_verifier_identity",
            "isolation_requirements", "oracle", "warmups", "measurements", "thermal_blocks", "statistics",
        },
        label="receipt.evaluator_bundle",
    )
    if data["schema"] != "auto_mlx.observation_bundle.v3":
        raise ContractError("evaluator bundle schema is incompatible", code=FailureCode.INVALID_VALUE)
    _string(data["candidate_id"], label="receipt.evaluator_bundle.candidate_id")
    validate_sha256(data["workload_hash"])
    RuntimeIdentity.from_dict(data["runtime"])
    if EvaluationPolicy.from_dict(data["policy"]) != policy:
        raise ContractError("evaluator bundle policy does not match receipt policy", code=FailureCode.INVALID_POLICY)
    thermal_blocks_wire = data["thermal_blocks"]
    if type(thermal_blocks_wire) is not list:
        raise ContractError("evaluator bundle thermal_blocks must be an array", code=FailureCode.WRONG_TYPE)
    # Wave B sequential sampling: the actual block count varies between
    # policy.measurement_runs (the starting/minimum count) and
    # policy.max_measurement_runs (the cap).  One thermal preflight entry
    # exists per block regardless of outcome (see Evaluator.evaluate), so
    # its length is the real, evidentiary block count for this receipt.
    block_count = len(thermal_blocks_wire)
    if not (policy.measurement_runs <= block_count <= policy.max_measurement_runs):
        raise ContractError("evaluator bundle thermal_blocks do not match measurement policy", code=FailureCode.INVALID_POLICY)
    for index, entry in enumerate(thermal_blocks_wire):
        entry_data = _validate_thermal_block_shape(entry, label=f"receipt.evaluator_bundle.thermal_blocks[{index}]")
        expected_block_id = f"block-{index + 1:04d}"
        if entry_data["block_id"] != expected_block_id or entry_data["block_index"] != index:
            raise ContractError("thermal block identity does not match measurement block order", code=FailureCode.IDENTITY_MISMATCH)
        if entry_data["policy"] != policy.thermal_gate_policy:
            raise ContractError("thermal block policy does not match evaluation policy", code=FailureCode.INVALID_POLICY)
    for name in ("baseline_runner_id", "candidate_runner_id"):
        _string(data[name], label=f"receipt.evaluator_bundle.{name}")
    for name in ("baseline_runner_digest", "candidate_runner_digest"):
        validate_sha256(data[name])
    if type(data["isolation_provider_id"]) not in {str, type(None)}:
        raise ContractError("receipt.evaluator_bundle.isolation_provider_id is invalid", code=FailureCode.WRONG_TYPE)
    if type(data["isolation_identity"]) not in {str, type(None)}:
        raise ContractError("receipt.evaluator_bundle.isolation_identity is invalid", code=FailureCode.WRONG_TYPE)
    if type(data["isolation_verifier_id"]) not in {str, type(None)}:
        raise ContractError("receipt.evaluator_bundle.isolation_verifier_id is invalid", code=FailureCode.WRONG_TYPE)
    if type(data["isolation_verifier_identity"]) not in {str, type(None)}:
        raise ContractError("receipt.evaluator_bundle.isolation_verifier_identity is invalid", code=FailureCode.WRONG_TYPE)
    if data["isolation_requirements"] is not None:
        if type(data["isolation_requirements"]) is not list or any(
            type(item) is not str or not item for item in data["isolation_requirements"]
        ):
            raise ContractError("receipt.evaluator_bundle.isolation_requirements is invalid", code=FailureCode.WRONG_TYPE)
        isolation_requirements = frozenset(data["isolation_requirements"])
    else:
        isolation_requirements = None
    isolation_values = (
        data["isolation_provider_id"],
        data["isolation_identity"],
        data["isolation_verifier_id"],
        data["isolation_verifier_identity"],
        isolation_requirements,
    )
    if any(value is not None for value in isolation_values) and not all(value is not None for value in isolation_values):
        raise ContractError("evaluator isolation provenance must be complete or absent", code=FailureCode.INVALID_VALUE)
    if isolation_requirements is not None and not _REQUIRED_ISOLATION.issubset(isolation_requirements):
        raise ContractError("evaluator isolation requirements are incomplete", code=FailureCode.SANDBOX_UNAVAILABLE)
    isolation_complete = all(value is not None for value in isolation_values)
    if isolation_complete:
        _string(data["isolation_provider_id"], label="receipt.evaluator_bundle.isolation_provider_id")
        validate_sha256(data["isolation_identity"])
        _string(data["isolation_verifier_id"], label="receipt.evaluator_bundle.isolation_verifier_id")
        validate_sha256(data["isolation_verifier_identity"])

    oracle_provenance = data["oracle"]
    if oracle_provenance is not None:
        oracle_data = _object(oracle_provenance, label="receipt.evaluator_bundle.oracle")
        _exact(oracle_data, {"label", "expected_digest", "expected_size", "expected_b64"}, label="receipt.evaluator_bundle.oracle")
        _string(oracle_data["label"], label="receipt.evaluator_bundle.oracle.label")
        validate_sha256(oracle_data["expected_digest"])
        expected = _decode_bytes(oracle_data["expected_b64"], label="receipt.evaluator_bundle.oracle.expected_b64")
        _integer(oracle_data["expected_size"], label="receipt.evaluator_bundle.oracle.expected_size", minimum=0)
        if len(expected) != oracle_data["expected_size"] or hashlib.sha256(expected).hexdigest() != oracle_data["expected_digest"]:
            raise ContractError("oracle provenance bytes do not match its digest", code=FailureCode.ORACLE_MISMATCH)

    warmups = data["warmups"]
    if type(warmups) is not list:
        raise ContractError("evaluator bundle warmups must be an array", code=FailureCode.WRONG_TYPE)
    if len(warmups) != policy.warmup_runs * 2:
        raise ContractError("evaluator bundle warmups are incomplete", code=FailureCode.INVALID_VALUE)
    warmup_success = True
    seen_arms: set = set()
    for index, warmup in enumerate(warmups):
        warmup_data = _object(warmup, label=f"receipt.evaluator_bundle.warmups[{index}]")
        _exact(
            warmup_data,
            {"sample_id", "arm", "record", "oracle", "warm_state", "sample_timing"},
            label=f"receipt.evaluator_bundle.warmups[{index}]",
        )
        _string(warmup_data["sample_id"], label=f"receipt.evaluator_bundle.warmups[{index}].sample_id")
        expected_warmup_arm = "baseline" if index % 2 == 0 else "candidate"
        if warmup_data["arm"] != expected_warmup_arm:
            raise ContractError("warmup arm is invalid", code=FailureCode.INVALID_VALUE)
        expected_warmup_id = f"warmup-{index // 2 + 1:04d}-{expected_warmup_arm}"
        if warmup_data["sample_id"] != expected_warmup_id:
            raise ContractError("warmup identity is not in evaluator order", code=FailureCode.IDENTITY_MISMATCH)
        stdout, stderr, record_data = _validate_record_wire(warmup_data["record"], label=f"receipt.evaluator_bundle.warmups[{index}].record")
        _validate_warm_state_shape(warmup_data["warm_state"], label=f"receipt.evaluator_bundle.warmups[{index}].warm_state")
        expected_warm_state = _warm_state_for(stderr, arm=warmup_data["arm"], seen_arms=seen_arms)
        if warmup_data["warm_state"] != expected_warm_state:
            raise ContractError("warmup warm_state was not independently recomputed", code=FailureCode.IDENTITY_MISMATCH)
        _validate_sample_timing_shape(warmup_data["sample_timing"], label=f"receipt.evaluator_bundle.warmups[{index}].sample_timing")
        expected_sample_timing = _sample_timing_for(
            runner_elapsed_ns=record_data["runner_elapsed_ns"], parent_elapsed_ns=record_data["parent_elapsed_ns"], stderr=stderr
        )
        if warmup_data["sample_timing"] != expected_sample_timing:
            raise ContractError("warmup sample_timing was not independently recomputed", code=FailureCode.IDENTITY_MISMATCH)
        expected_runner = data["baseline_runner_id"] if warmup_data["arm"] == "baseline" else data["candidate_runner_id"]
        expected_digest = data["baseline_runner_digest"] if warmup_data["arm"] == "baseline" else data["candidate_runner_digest"]
        if (
            record_data["candidate_id"] != data["candidate_id"]
            or record_data["workload_hash"] != data["workload_hash"]
            or record_data["runner_id"] != expected_runner
            or record_data["runner_digest"] != expected_digest
            or record_data["arm"] != warmup_data["arm"]
            or record_data["observation_id"] != warmup_data["sample_id"]
        ):
            raise ContractError("warmup record provenance does not match its slot", code=FailureCode.IDENTITY_MISMATCH)
        if record_data["isolation"] is None:
            warmup_success = False
        elif (
            not isolation_complete
            or record_data["isolation"]["provider_id"] != data["isolation_provider_id"]
            or record_data["isolation"]["identity"] != data["isolation_identity"]
            or record_data["isolation"]["verifier_id"] != data["isolation_verifier_id"]
            or record_data["isolation"]["verifier_identity"] != data["isolation_verifier_identity"]
        ):
            raise ContractError("warmup isolation provenance does not match bundle", code=FailureCode.IDENTITY_MISMATCH)
        elif not isolation_requirements.issubset(record_data["isolation"]["requirements"]):
            raise ContractError("warmup isolation requirements do not match plan", code=FailureCode.IDENTITY_MISMATCH)
        if warmup_data["oracle"] is not None:
            matched, expected_digest, _ = _validate_oracle_wire(warmup_data["oracle"], stdout, label=f"receipt.evaluator_bundle.warmups[{index}].oracle")
            if expected_digest != oracle_provenance["expected_digest"]:
                raise ContractError("warmup oracle is not bound to bundle oracle", code=FailureCode.IDENTITY_MISMATCH)
            runner_elapsed = record_data["runner_elapsed_ns"]
            if (
                record_data["status"] != "success"
                or runner_elapsed is None
                or runner_elapsed <= 0
                or record_data["output_truncated"]
                or not matched
            ):
                warmup_success = False
        else:
            raise ContractError("warmup oracle metadata is missing", code=FailureCode.ORACLE_MISMATCH)

    measurements = _object(data["measurements"], label="receipt.evaluator_bundle.measurements")
    _exact(measurements, {"plan_digest", "accepted", "rejection_reasons", "unexpected_sample_ids", "blocks"}, label="receipt.evaluator_bundle.measurements")
    validate_sha256(measurements["plan_digest"])
    _boolean(measurements["accepted"], label="receipt.evaluator_bundle.measurements.accepted")
    for name in ("rejection_reasons", "unexpected_sample_ids"):
        if type(measurements[name]) is not list or any(type(item) is not str for item in measurements[name]):
            raise ContractError(f"receipt.evaluator_bundle.measurements.{name} must be string arrays", code=FailureCode.WRONG_TYPE)
    blocks = measurements["blocks"]
    if type(blocks) is not list or len(blocks) != block_count:
        raise ContractError("evaluator bundle blocks do not match measurement policy", code=FailureCode.INVALID_POLICY)
    from .oracle import ExactOutputOracle
    from .measurement import PairedMeasurementPlan

    if oracle_provenance is None:
        raise ContractError("evaluator oracle provenance is missing", code=FailureCode.ORACLE_MISMATCH)
    expected_oracle = ExactOutputOracle(
        _decode_bytes(oracle_provenance["expected_b64"], label="receipt.evaluator_bundle.oracle.expected_b64"),
        label=oracle_provenance["label"],
        expected_digest=oracle_provenance["expected_digest"],
    )
    expected_plan = PairedMeasurementPlan.create(
        len(blocks),
        candidate_id=data["candidate_id"],
        workload_hash=data["workload_hash"],
        baseline_runner_id=data["baseline_runner_id"],
        baseline_runner_digest=data["baseline_runner_digest"],
        candidate_runner_id=data["candidate_runner_id"],
        candidate_runner_digest=data["candidate_runner_digest"],
        oracle=expected_oracle,
        require_isolation=True,
        isolation_provider_id=data["isolation_provider_id"],
        isolation_identity=data["isolation_identity"],
        isolation_verifier_id=data["isolation_verifier_id"],
        isolation_verifier_identity=data["isolation_verifier_identity"],
        isolation_requirements=isolation_requirements,
    )
    if expected_plan.plan_digest != measurements["plan_digest"]:
        raise ContractError("measurement plan digest does not match its bound provenance", code=FailureCode.IDENTITY_MISMATCH)
    baseline_values: list[int] = []
    candidate_values: list[int] = []
    drift_values: list[int] = []
    outcomes: list[dict[str, Any]] = []
    all_slots = 0
    all_success = True
    block_baseline_points: list[list[int]] = []
    block_candidate_points: list[list[int]] = []
    for block_index, block in enumerate(blocks):
        block_data = _object(block, label=f"receipt.evaluator_bundle.measurements.blocks[{block_index}]")
        _exact(block_data, {"block_id", "block_index", "sequence", "accepted", "rejection_reasons", "samples", "dispersion_inputs"}, label=f"receipt.evaluator_bundle.measurements.blocks[{block_index}]")
        if block_data["block_index"] != block_index or type(block_data["sequence"]) is not list or len(block_data["sequence"]) != 4:
            raise ContractError("measurement block ordering is invalid", code=FailureCode.INVALID_VALUE)
        sequence = tuple(block_data["sequence"])
        if sequence not in {("baseline", "candidate", "candidate", "baseline"), ("candidate", "baseline", "baseline", "candidate")}:
            raise ContractError("measurement block is not ABBA or BAAB", code=FailureCode.INVALID_VALUE)
        _string(block_data["block_id"], label=f"block[{block_index}].block_id")
        _boolean(block_data["accepted"], label=f"block[{block_index}].accepted")
        if type(block_data["rejection_reasons"]) is not list or any(type(item) is not str for item in block_data["rejection_reasons"]):
            raise ContractError("measurement block rejection reasons are invalid", code=FailureCode.WRONG_TYPE)
        samples = block_data["samples"]
        if type(samples) is not list or len(samples) != 4:
            raise ContractError("every measurement block must retain four arm-specific slots", code=FailureCode.INVALID_VALUE)
        dispersion = _object(block_data["dispersion_inputs"], label=f"block[{block_index}].dispersion_inputs")
        _exact(
            dispersion,
            {
                "ordered_parent_elapsed_ns", "baseline_elapsed_ns", "candidate_elapsed_ns", "baseline_drift_ns",
                "ordered_runner_elapsed_ns", "baseline_runner_elapsed_ns", "candidate_runner_elapsed_ns",
            },
            label=f"block[{block_index}].dispersion_inputs",
        )
        for name in (
            "ordered_parent_elapsed_ns", "baseline_elapsed_ns", "candidate_elapsed_ns",
            "ordered_runner_elapsed_ns", "baseline_runner_elapsed_ns", "candidate_runner_elapsed_ns",
        ):
            if type(dispersion[name]) is not list or any(value is not None and (type(value) is not int or value < 0) for value in dispersion[name]):
                raise ContractError(f"block[{block_index}].dispersion_inputs.{name} is invalid", code=FailureCode.WRONG_TYPE)
        if len(dispersion["ordered_parent_elapsed_ns"]) != 4 or len(dispersion["baseline_elapsed_ns"]) != 2 or len(dispersion["candidate_elapsed_ns"]) != 2:
            raise ContractError("dispersion inputs do not retain all four arm-specific slots", code=FailureCode.INVALID_VALUE)
        if (
            len(dispersion["ordered_runner_elapsed_ns"]) != 4
            or len(dispersion["baseline_runner_elapsed_ns"]) != 2
            or len(dispersion["candidate_runner_elapsed_ns"]) != 2
        ):
            raise ContractError("runner dispersion inputs do not retain all four arm-specific slots", code=FailureCode.INVALID_VALUE)
        expected_baseline = [value for value, arm in zip(dispersion["ordered_parent_elapsed_ns"], sequence) if arm == "baseline"]
        expected_candidate = [value for value, arm in zip(dispersion["ordered_parent_elapsed_ns"], sequence) if arm == "candidate"]
        if dispersion["baseline_elapsed_ns"] != expected_baseline or dispersion["candidate_elapsed_ns"] != expected_candidate:
            raise ContractError("dispersion inputs do not match arm ordering", code=FailureCode.IDENTITY_MISMATCH)
        expected_runner_baseline = [value for value, arm in zip(dispersion["ordered_runner_elapsed_ns"], sequence) if arm == "baseline"]
        expected_runner_candidate = [value for value, arm in zip(dispersion["ordered_runner_elapsed_ns"], sequence) if arm == "candidate"]
        if (
            dispersion["baseline_runner_elapsed_ns"] != expected_runner_baseline
            or dispersion["candidate_runner_elapsed_ns"] != expected_runner_candidate
        ):
            raise ContractError("runner dispersion inputs do not match arm ordering", code=FailureCode.IDENTITY_MISMATCH)
        if any(value is None for value in expected_baseline):
            block_drift = 0
        else:
            block_drift = abs(expected_baseline[1] - expected_baseline[0])
        if dispersion["baseline_drift_ns"] != (expected_baseline[1] - expected_baseline[0] if all(value is not None for value in expected_baseline) else None):
            raise ContractError("baseline drift was not independently recomputed", code=FailureCode.IDENTITY_MISMATCH)
        drift_values.append(block_drift)
        block_baseline_this: list[int] = []
        block_candidate_this: list[int] = []
        for slot_index, sample in enumerate(samples):
            sample_data = _object(sample, label=f"block[{block_index}].samples[{slot_index}]")
            _exact(
                sample_data,
                {"sample_id", "block_id", "slot_index", "arm", "record", "oracle", "warm_state", "sample_timing"},
                label=f"block[{block_index}].samples[{slot_index}]",
            )
            _string(sample_data["sample_id"], label=f"block[{block_index}].sample_id")
            expected_slot = expected_plan.blocks[block_index].slots[slot_index]
            if (
                sample_data["sample_id"] != expected_slot.sample_id
                or sample_data["block_id"] != block_data["block_id"]
                or block_data["block_id"] != expected_plan.blocks[block_index].block_id
                or sample_data["slot_index"] != slot_index
                or sample_data["arm"] != sequence[slot_index]
            ):
                raise ContractError("measurement slot identity does not match its ABBA/BAAB plan", code=FailureCode.IDENTITY_MISMATCH)
            _validate_warm_state_shape(sample_data["warm_state"], label=f"block[{block_index}].samples[{slot_index}].warm_state")
            _validate_sample_timing_shape(sample_data["sample_timing"], label=f"block[{block_index}].samples[{slot_index}].sample_timing")
            if sample_data["record"] is None or sample_data["oracle"] is None:
                if sample_data["record"] is None and sample_data["oracle"] is not None:
                    raise ContractError("missing measurement record cannot carry oracle metadata", code=FailureCode.INVALID_VALUE)
                expected_warm_state = _warm_state_for(None, arm=sample_data["arm"], seen_arms=seen_arms)
                if sample_data["warm_state"] != expected_warm_state:
                    raise ContractError("measurement warm_state was not independently recomputed", code=FailureCode.IDENTITY_MISMATCH)
                expected_sample_timing = _sample_timing_for(runner_elapsed_ns=None, parent_elapsed_ns=1, stderr=None)
                if sample_data["sample_timing"] != expected_sample_timing:
                    raise ContractError("measurement sample_timing was not independently recomputed", code=FailureCode.IDENTITY_MISMATCH)
                all_success = False
                continue
            stdout, stderr, record_data = _validate_record_wire(sample_data["record"], label=f"block[{block_index}].samples[{slot_index}].record")
            expected_warm_state = _warm_state_for(stderr, arm=sample_data["arm"], seen_arms=seen_arms)
            if sample_data["warm_state"] != expected_warm_state:
                raise ContractError("measurement warm_state was not independently recomputed", code=FailureCode.IDENTITY_MISMATCH)
            expected_sample_timing = _sample_timing_for(
                runner_elapsed_ns=record_data["runner_elapsed_ns"], parent_elapsed_ns=record_data["parent_elapsed_ns"], stderr=stderr
            )
            if sample_data["sample_timing"] != expected_sample_timing:
                raise ContractError("measurement sample_timing was not independently recomputed", code=FailureCode.IDENTITY_MISMATCH)
            if (
                record_data["candidate_id"] != data["candidate_id"]
                or record_data["workload_hash"] != data["workload_hash"]
                or record_data["runner_id"] != (data["baseline_runner_id"] if sample_data["arm"] == "baseline" else data["candidate_runner_id"])
                or record_data["runner_digest"] != (data["baseline_runner_digest"] if sample_data["arm"] == "baseline" else data["candidate_runner_digest"])
                or record_data["observation_id"] != sample_data["sample_id"]
                or record_data["arm"] != sample_data["arm"]
            ):
                raise ContractError("measurement record provenance does not match its slot", code=FailureCode.IDENTITY_MISMATCH)
            matched, expected_digest, _ = _validate_oracle_wire(sample_data["oracle"], stdout, label=f"block[{block_index}].samples[{slot_index}].oracle")
            if expected_digest != oracle_provenance["expected_digest"]:
                raise ContractError("measurement oracle is not bound to bundle oracle", code=FailureCode.IDENTITY_MISMATCH)
            if record_data["isolation"] is None:
                all_success = False
            elif not isolation_complete:
                raise ContractError("measurement carries isolation proof without bundle provenance", code=FailureCode.IDENTITY_MISMATCH)
            else:
                isolation_data = record_data["isolation"]
                if (
                    isolation_data["provider_id"] != data["isolation_provider_id"]
                    or isolation_data["identity"] != data["isolation_identity"]
                    or isolation_data["verifier_id"] != data["isolation_verifier_id"]
                    or isolation_data["verifier_identity"] != data["isolation_verifier_identity"]
                ):
                    raise ContractError("measurement isolation provenance does not match bundle", code=FailureCode.IDENTITY_MISMATCH)
                if not isolation_requirements.issubset(isolation_data["requirements"]):
                    raise ContractError("measurement isolation requirements do not match plan", code=FailureCode.IDENTITY_MISMATCH)
            status = record_data["status"]
            # The evidentiary quantity for gain math is the runner span, not
            # the full-sample span -- see auto_mlx.executor.execute_plan.
            elapsed = record_data["runner_elapsed_ns"]
            if status != "success" or elapsed is None or elapsed <= 0 or record_data["output_truncated"] or not matched:
                all_success = False
            if elapsed is not None:
                if sample_data["arm"] == "baseline":
                    baseline_values.append(elapsed)
                    block_baseline_this.append(sample_data["sample_timing"]["point_estimate_ns"])
                else:
                    candidate_values.append(elapsed)
                    block_candidate_this.append(sample_data["sample_timing"]["point_estimate_ns"])
            outcomes.append({
                "index": all_slots,
                "actual_sha256": hashlib.sha256(stdout).hexdigest(),
                "oracle_sha256": sample_data["oracle"]["expected_digest"],
                "match": matched,
            })
            all_slots += 1
        block_baseline_points.append(block_baseline_this)
        block_candidate_points.append(block_candidate_this)
    expected_slots = block_count * 4
    compatible = (
        warmup_success
        and all_success
        and all_slots == expected_slots
        and len(baseline_values) == block_count * 2
        and len(candidate_values) == block_count * 2
    )
    candidate = _stats(candidate_values)
    baseline = _stats(baseline_values)
    matches = sum(1 for outcome in outcomes if outcome["match"])
    gain = baseline["sum_ns"] - candidate["sum_ns"]

    if compatible:
        if data["statistics"] is None:
            raise ContractError("evaluator bundle statistics is missing for a compatible measurement set", code=FailureCode.INVALID_VALUE)
        stored_statistics = StatisticsVerdict.from_dict(data["statistics"])
        expected_statistics = compute_statistics_verdict(
            block_baseline_points=block_baseline_points,
            block_candidate_points=block_candidate_points,
            k_repetitions=policy.k_repetitions,
            measurement_runs=policy.measurement_runs,
            max_measurement_runs=policy.max_measurement_runs,
            min_effect_bps=policy.min_effect_bps,
            bootstrap_resamples=policy.bootstrap_resamples,
            bootstrap_seed=stored_statistics.bootstrap_seed,
            calibration=policy.calibration,
        )
        if expected_statistics.to_dict() != data["statistics"]:
            raise ContractError("evaluator bundle statistics were not independently recomputed", code=FailureCode.IDENTITY_MISMATCH)
        # Legitimacy of stopping: an inconclusive verdict is only a
        # legitimate stopping point at the sequential sampling cap --
        # otherwise sequential extension should have continued (see
        # Evaluator.evaluate).  A decisive verdict may stop at any block
        # count in [measurement_runs, max_measurement_runs].
        #
        # Wave C (auto_mlx.tuning) exception: a policy explicitly marked
        # ``racing`` may ALSO stop on an inconclusive verdict before the cap
        # when the CI upper bound has already fallen below the min-effect
        # threshold -- i.e. this candidate cannot become a winner no matter
        # how many further blocks are added, so continuing to the cap would
        # only spend budget confirming a foregone conclusion. This is
        # independently re-verified here from the same recomputed CI/
        # threshold every other statistics field is checked against, not
        # merely trusted from the stored verdict -- a bug or forged receipt
        # cannot claim an illegitimate early stop this way.
        racing_eliminable = (
            policy.racing and expected_statistics.ci_upper_ns < expected_statistics.min_effect_ns
        )
        if (
            expected_statistics.verdict == VERDICT_INCONCLUSIVE
            and block_count != policy.max_measurement_runs
            and not racing_eliminable
        ):
            raise ContractError(
                "evaluation stopped on an inconclusive verdict before reaching the sequential sampling cap",
                code=FailureCode.INVALID_VALUE,
            )
        statistics_out: dict[str, Any] | None = dict(data["statistics"])
    elif data["statistics"] is not None:
        raise ContractError("evaluator bundle statistics must be null for an incomplete measurement set", code=FailureCode.INVALID_VALUE)
    else:
        statistics_out = None

    return {
        "aggregates": {"candidate": candidate, "baseline": baseline},
        "oracle": {"sample_count": all_slots, "matches": matches, "mismatches": all_slots - matches, "all_match": all_slots == matches and all_slots == expected_slots, "outcomes": outcomes},
        "compatibility": {
            "warmup_runs": policy.warmup_runs,
            "measurement_runs": policy.measurement_runs,
            "observed_samples": all_slots,
            "compatible": compatible,
            "thermal_blocks": thermal_blocks_wire,
            "thermal_gate_policy": policy.thermal_gate_policy,
            "block_count_used": block_count,
        },
        "statistics": statistics_out,
        "metrics": {
            "drift": {"count": len(drift_values), "sum": sum(drift_values), "min": min(drift_values) if drift_values else 0, "max": max(drift_values) if drift_values else 0},
            "dispersion": {"candidate": _mean_abs_deviation(candidate_values), "baseline": _mean_abs_deviation(baseline_values), "candidate_range_ns": candidate["range_ns"], "baseline_range_ns": baseline["range_ns"]},
            "gain": {"baseline_sum_ns": baseline["sum_ns"], "candidate_sum_ns": candidate["sum_ns"], "delta_ns": gain, "improved": gain > 0, "numerator": gain, "denominator": baseline["sum_ns"] if baseline["sum_ns"] else 1},
        },
    }


def _recompute_evaluator_bundle_fields(value: Mapping[str, Any], policy: EvaluationPolicy) -> dict[str, Any]:
    """Independent derived-field recomputation for the evaluator-owned wire adapter."""

    return _validate_evaluator_bundle_wire(value, policy)


def _raw_samples_match_evaluator_bundle(
    raw_samples: Sequence[RawSample], bundle: Mapping[str, Any]
) -> bool:
    """Bind the compact raw-sample lane to every retained evaluator record."""

    try:
        expected: list[dict[str, Any]] = []
        for block in bundle["measurements"]["blocks"]:
            for sample in block["samples"]:
                record = sample["record"]
                stdout = b"" if record is None else _decode_bytes(record["stdout_b64"], label="raw evaluator stdout")
                expected.append(
                    RawSample(
                        len(expected),
                        0 if record is None else record["parent_elapsed_ns"],
                        0 if record is None else record["parent_elapsed_ns"],
                        {"stdout_b64": _encode_bytes(stdout)},
                        {"oracle": None if sample["oracle"] is None else sample["oracle"]},
                        0,
                    ).to_dict()
                )
        return [sample.to_dict() for sample in raw_samples] == expected
    except (ContractError, KeyError, TypeError):
        return False


def _receipt_from_observation_bundle(
    bundle: Any,
    workload: FrozenWorkload | Mapping[str, Any],
    candidate: CandidateProposal | Mapping[str, Any],
    policy: EvaluationPolicy | Mapping[str, Any],
    *,
    oracle: Any | None,
    created_at_ns: int | None,
) -> Receipt:
    from .evaluator import ObservationBundle

    if type(bundle) is not ObservationBundle:
        raise ContractError("receipt adapter requires the evaluator ObservationBundle", code=FailureCode.WRONG_TYPE)
    frozen_workload = _coerce_workload(workload)
    frozen_candidate = _coerce_candidate(candidate, frozen_workload)
    frozen_policy = _coerce_policy(policy)
    from .evaluator import _execution_policy_digest, _execution_policy_from_contract
    from .executor import ExecutionPolicy

    bundle_policy = getattr(bundle, "evaluation_policy", None)
    if type(bundle_policy) is not EvaluationPolicy or bundle_policy != frozen_policy:
        raise ContractError(
            "evaluator bundle policy does not exactly match receipt policy",
            code=FailureCode.INVALID_POLICY,
        )
    expected_policy_digest = sha256_hex(frozen_policy.to_dict())
    if type(getattr(bundle, "policy_digest", None)) is not str or bundle.policy_digest != expected_policy_digest:
        raise ContractError(
            "evaluator bundle policy digest does not match receipt policy",
            code=FailureCode.INVALID_POLICY,
        )
    bundle_execution_policy = getattr(bundle, "execution_policy", None)
    expected_execution_policy = _execution_policy_from_contract(frozen_policy)
    if type(bundle_execution_policy) is not ExecutionPolicy or bundle_execution_policy.to_dict() != expected_execution_policy.to_dict():
        raise ContractError(
            "evaluator bundle execution policy does not match receipt policy",
            code=FailureCode.INVALID_POLICY,
        )
    expected_execution_policy_digest = _execution_policy_digest(bundle_execution_policy)
    if type(getattr(bundle, "execution_policy_digest", None)) is not str or bundle.execution_policy_digest != expected_execution_policy_digest:
        raise ContractError(
            "evaluator bundle execution policy digest does not match receipt policy",
            code=FailureCode.INVALID_POLICY,
        )
    if bundle.candidate_id != frozen_candidate.candidate_id or bundle.workload_hash != frozen_workload.workload_hash:
        raise ContractError("evaluator bundle identity does not match receipt context", code=FailureCode.IDENTITY_MISMATCH)
    from .oracle import ExactOutputOracle

    if not isinstance(bundle.runtime, RuntimeIdentity):
        raise ContractError("evaluator bundle runtime is invalid", code=FailureCode.INVALID_RUNTIME)
    if not isinstance(oracle, ExactOutputOracle):
        raise ContractError("receipt adapter requires the evaluator ExactOutputOracle", code=FailureCode.ORACLE_MISMATCH)
    recomputed_measurements = bundle.recompute_measurements()
    evaluator_bundle = _observation_bundle_to_wire(
        bundle,
        frozen_policy,
        oracle,
        measurements=recomputed_measurements,
    )
    raw_samples: list[RawSample] = []
    for block in recomputed_measurements.blocks:
        for sample in block.samples:
            record = sample.record
            stdout = b"" if record is None else record.stdout
            raw_samples.append(
                RawSample(
                    len(raw_samples),
                    0 if record is None else record.parent_elapsed_ns,
                    0 if record is None else record.parent_elapsed_ns,
                    {"stdout_b64": _encode_bytes(stdout)},
                    {"oracle": None if sample.oracle is None else sample.oracle.to_dict()},
                    0,
                )
            )
    # Status reflects the bundle's OWN, independently-recomputed evidence
    # judgment (ObservationBundle.accepted / .promotion_eligible in
    # evaluator.py) -- never an unconditional hold.  This is still only the
    # G1 evidence layer's opinion: G2 promotion (make_promotion_decision)
    # separately and independently re-derives everything from raw_samples,
    # and activation additionally requires a real out-of-band supervisor
    # attestation (see auto_mlx.supervisor) before a receipt can ever
    # activate -- a "complete" status here is necessary, never sufficient.
    bundle_accepted = bundle.accepted
    bundle_promotion_eligible = bundle.promotion_eligible
    if bundle_accepted and bundle_promotion_eligible:
        status = "complete"
        failure = None
    else:
        status = "failed"
        reasons = list(recomputed_measurements.rejection_reasons)
        if not bundle_accepted and not reasons:
            reasons.append("bundle_not_accepted")
        if bundle_accepted and not bundle_promotion_eligible:
            reasons.append("bundle_not_promotion_eligible")
        failure = Failure(
            FailureCode.RUNTIME_FAILURE,
            "G0 evaluator observation bundle is not accepted, promotion-eligible evidence",
            {"reasons": reasons},
        )
    return Receipt(
        frozen_workload,
        frozen_candidate,
        frozen_policy,
        bundle.runtime,
        raw_samples,
        artifacts=frozen_workload.artifacts,
        created_at_ns=created_at_ns,
        failure=failure,
        evaluator_bundle=evaluator_bundle,
        status=status,
    )


receipt_from_observation_bundle = _receipt_from_observation_bundle
adapt_observation_bundle = _receipt_from_observation_bundle


_RECEIPT_FIELDS: Final = {
    "schema",
    "provenance",
    "workload",
    "candidate",
    "policy",
    "runtime",
    "artifacts",
    "raw_samples",
    "created_at_ns",
    "status",
    "aggregates",
    "oracle",
    "compatibility",
    "metrics",
    "statistics",
    "failure",
    "evaluator_bundle",
    "receipt_id",
}


@dataclass(frozen=True, slots=True, init=False)
class Receipt:
    """A closed, immutable evaluation receipt."""

    workload: FrozenWorkload
    candidate: CandidateProposal
    policy: EvaluationPolicy
    runtime: RuntimeIdentity
    artifacts: tuple[Artifact, ...]
    raw_samples: tuple[RawSample, ...]
    created_at_ns: int
    status: str
    aggregates: Mapping[str, Any]
    oracle: Mapping[str, Any]
    compatibility: Mapping[str, Any]
    metrics: Mapping[str, Any]
    statistics: Mapping[str, Any] | None
    failure: Failure | None
    evaluator_bundle: Mapping[str, Any] | None
    receipt_id: str

    def __init__(
        self,
        workload: FrozenWorkload | Mapping[str, Any],
        candidate: CandidateProposal | Mapping[str, Any],
        policy: EvaluationPolicy | Mapping[str, Any],
        runtime: RuntimeIdentity | Mapping[str, Any],
        raw_samples: Sequence[RawSample | Mapping[str, Any]],
        *,
        artifacts: Sequence[Artifact | Mapping[str, Any]] | None = None,
        created_at_ns: int | None = None,
        failure: Failure | Mapping[str, Any] | None = None,
        aggregates: Mapping[str, Any] | None = None,
        oracle: Mapping[str, Any] | None = None,
        compatibility: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        # Sentinel default (not None -- see _MISSING_STATISTICS above):
        # distinguishes "caller omitted this" (use the recomputed value,
        # which may itself legitimately be None) from "caller passed None".
        statistics: Any = _MISSING_STATISTICS,
        status: str | None = None,
        evaluator_bundle: Mapping[str, Any] | None = None,
        receipt_id: str | None = None,
    ) -> None:
        frozen_workload = _coerce_workload(workload)
        frozen_candidate = _coerce_candidate(candidate, frozen_workload)
        frozen_policy = _coerce_policy(policy)
        frozen_runtime = _coerce_runtime(runtime)
        if type(raw_samples) not in {list, tuple}:
            raise ContractError("raw_samples must be an array", code=FailureCode.WRONG_TYPE)
        samples = tuple(item if isinstance(item, RawSample) else RawSample.from_dict(item) for item in raw_samples)
        if artifacts is None:
            frozen_artifacts = tuple(frozen_workload.artifacts)
        else:
            if type(artifacts) not in {list, tuple}:
                raise ContractError("artifacts must be an array", code=FailureCode.WRONG_TYPE)
            frozen_artifacts = tuple(
                item if isinstance(item, Artifact) else Artifact.from_dict(item) for item in artifacts
            )
        timestamp = time.time_ns() if created_at_ns is None else _integer(created_at_ns, label="created_at_ns", minimum=0)
        frozen_failure = _coerce_failure(failure)
        computed = recompute_receipt_fields(samples, frozen_policy, evaluator_bundle=evaluator_bundle)
        body_aggregates = dict(computed["aggregates"] if aggregates is None else aggregates)
        body_oracle = dict(computed["oracle"] if oracle is None else oracle)
        body_compatibility = dict(computed["compatibility"] if compatibility is None else compatibility)
        body_metrics = dict(computed["metrics"] if metrics is None else metrics)
        body_statistics = computed["statistics"] if statistics is _MISSING_STATISTICS else statistics
        if body_statistics is not None:
            body_statistics = dict(body_statistics)
        if status is None:
            body_status = "failed" if frozen_failure is not None else "complete"
        else:
            body_status = _string(status, label="status")
            if body_status not in {"complete", "failed"}:
                raise ContractError("status must be complete or failed", code=FailureCode.INVALID_VALUE)
        for label, value in (
            ("aggregates", body_aggregates),
            ("oracle", body_oracle),
            ("compatibility", body_compatibility),
            ("metrics", body_metrics),
        ):
            _json_value(value, label=label)
        if body_statistics is not None:
            _json_value(body_statistics, label="statistics")
        _validate_derived_fields(body_aggregates, body_oracle, body_compatibility, body_metrics, body_statistics)
        if evaluator_bundle is not None:
            _json_value(evaluator_bundle, label="receipt.evaluator_bundle")
        frozen_aggregates = _freeze_json(body_aggregates, path="receipt.aggregates")
        frozen_oracle = _freeze_json(body_oracle, path="receipt.oracle")
        frozen_compatibility = _freeze_json(body_compatibility, path="receipt.compatibility")
        frozen_metrics = _freeze_json(body_metrics, path="receipt.metrics")
        frozen_statistics = None if body_statistics is None else _freeze_json(body_statistics, path="receipt.statistics")
        object.__setattr__(self, "workload", frozen_workload)
        object.__setattr__(self, "candidate", frozen_candidate)
        object.__setattr__(self, "policy", frozen_policy)
        object.__setattr__(self, "runtime", frozen_runtime)
        object.__setattr__(self, "artifacts", frozen_artifacts)
        object.__setattr__(self, "raw_samples", samples)
        object.__setattr__(self, "created_at_ns", timestamp)
        object.__setattr__(self, "status", body_status)
        object.__setattr__(self, "aggregates", frozen_aggregates)
        object.__setattr__(self, "oracle", frozen_oracle)
        object.__setattr__(self, "compatibility", frozen_compatibility)
        object.__setattr__(self, "metrics", frozen_metrics)
        object.__setattr__(self, "statistics", frozen_statistics)
        object.__setattr__(self, "failure", frozen_failure)
        object.__setattr__(self, "evaluator_bundle", None if evaluator_bundle is None else _freeze_json(evaluator_bundle, path="receipt.evaluator_bundle"))
        expected_id = sha256_hex(self._body_dict())
        if receipt_id is None:
            final_id = expected_id
        else:
            final_id = validate_sha256(receipt_id)
            if final_id != expected_id:
                raise ContractError("receipt_id does not match canonical receipt body", code=FailureCode.IDENTITY_MISMATCH)
        object.__setattr__(self, "receipt_id", final_id)

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "provenance": "local_evaluation",
            "workload": self.workload.to_dict(),
            "candidate": self.candidate.to_dict(),
            "policy": self.policy.to_dict(),
            "runtime": self.runtime.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "raw_samples": [sample.to_dict() for sample in self.raw_samples],
            "created_at_ns": self.created_at_ns,
            "status": self.status,
            "aggregates": _thaw_json(self.aggregates),
            "oracle": _thaw_json(self.oracle),
            "compatibility": _thaw_json(self.compatibility),
            "metrics": _thaw_json(self.metrics),
            "statistics": None if self.statistics is None else _thaw_json(self.statistics),
            "failure": self.failure.to_dict() if self.failure is not None else None,
            "evaluator_bundle": None if self.evaluator_bundle is None else _thaw_json(self.evaluator_bundle),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._body_dict()
        result["receipt_id"] = self.receipt_id
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def identity(self) -> str:
        return self.receipt_id

    @property
    def complete(self) -> bool:
        return self.status == "complete" and self.failure is None

    @classmethod
    def from_dict(cls, value: Any) -> "Receipt":
        data = _object(value, label="receipt")
        _exact(data, _RECEIPT_FIELDS, label="receipt")
        if data["schema"] != RECEIPT_SCHEMA or data["provenance"] != "local_evaluation":
            raise ContractError("receipt schema or provenance is incompatible", code=FailureCode.INVALID_VALUE)
        workload = FrozenWorkload.from_dict(data["workload"])
        candidate = CandidateProposal.from_dict(data["candidate"], workload)
        policy = EvaluationPolicy.from_dict(data["policy"])
        runtime = RuntimeIdentity.from_dict(data["runtime"])
        if type(data["artifacts"]) is not list:
            raise ContractError("receipt.artifacts must be an array", code=FailureCode.WRONG_TYPE)
        if type(data["raw_samples"]) is not list:
            raise ContractError("receipt.raw_samples must be an array", code=FailureCode.WRONG_TYPE)
        artifacts = tuple(Artifact.from_dict(item) for item in data["artifacts"])
        samples = tuple(RawSample.from_dict(item) for item in data["raw_samples"])
        if data["failure"] is not None and type(data["failure"]) is not dict:
            raise ContractError("receipt.failure must be an object or null", code=FailureCode.WRONG_TYPE)
        receipt = cls(
            workload,
            candidate,
            policy,
            runtime,
            samples,
            artifacts=artifacts,
            created_at_ns=data["created_at_ns"],
            failure=data["failure"],
            aggregates=data["aggregates"],
            oracle=data["oracle"],
            compatibility=data["compatibility"],
            metrics=data["metrics"],
            statistics=data["statistics"],
            status=data["status"],
            evaluator_bundle=data["evaluator_bundle"],
            receipt_id=data["receipt_id"],
        )
        if receipt.receipt_id != sha256_hex(receipt._body_dict()):
            raise ContractError("receipt_id does not match canonical receipt body", code=FailureCode.IDENTITY_MISMATCH)
        return receipt

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> "Receipt":
        return cls.from_dict(strict_json_loads(value))

    @classmethod
    def from_observation_bundle(
        cls,
        bundle: Any,
        workload: FrozenWorkload | Mapping[str, Any],
        candidate: CandidateProposal | Mapping[str, Any],
        policy: EvaluationPolicy | Mapping[str, Any],
        *,
        oracle: Any | None = None,
        created_at_ns: int | None = None,
    ) -> "Receipt":
        return _receipt_from_observation_bundle(
            bundle,
            workload,
            candidate,
            policy,
            oracle=oracle,
            created_at_ns=created_at_ns,
        )


@dataclass(frozen=True, slots=True)
class ReceiptValidation:
    """Validator output; promotion accepts this value, never a raw receipt."""

    receipt: Receipt | None
    valid: bool
    complete: bool
    local: bool
    failures: tuple[Failure, ...]
    recomputed: Mapping[str, Any]
    attested: bool = False
    attestation: str | None = None
    artifacts_verified: bool = False
    _proof: _ValidationProof | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "recomputed", _freeze_json(self.recomputed, path="validation.recomputed"))
        if self.attestation is not None:
            validate_sha256(self.attestation)

    @property
    def ok(self) -> bool:
        proof = self._proof
        return (
            self.valid
            and self.complete
            and self.local
            and self.attested
            and self.artifacts_verified
            and isinstance(proof, _ValidationProof)
            and proof.token is _VALIDATION_TOKEN
            and proof.receipt_id == self.receipt_id
            and proof.attestation == self.attestation
        )

    @property
    def receipt_id(self) -> str | None:
        return self.receipt.receipt_id if self.receipt is not None else None


def _failure(code: FailureCode, message: str, **details: Any) -> Failure:
    return Failure(code, message, details)


def validate_receipt(
    value: Receipt | Mapping[str, Any],
    *,
    workload: FrozenWorkload | Mapping[str, Any] | None = None,
    candidate: CandidateProposal | Mapping[str, Any] | None = None,
    policy: EvaluationPolicy | Mapping[str, Any] | None = None,
    runtime: RuntimeIdentity | Mapping[str, Any] | None = None,
    artifact_root: str | os.PathLike[str] | None = None,
    attestation: str | bytes | bytearray | None = None,
    attestation_key: bytes | bytearray | None = None,
) -> ReceiptValidation:
    """Recompute evidence; promotion additionally requires an out-of-band HMAC."""

    failures: list[Failure] = []
    try:
        receipt = value if isinstance(value, Receipt) else Receipt.from_dict(value)
    except ContractError as exc:
        return ReceiptValidation(
            None,
            False,
            False,
            False,
            (Failure(exc.code, str(exc)),),
            {},
            attested=False,
            _proof=_ValidationProof(_VALIDATION_TOKEN, None, None),
        )

    expected_body_id = sha256_hex(receipt._body_dict())
    if receipt.receipt_id != expected_body_id:
        failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "receipt_id does not match canonical receipt body"))
    if receipt.candidate.workload_hash != receipt.workload.workload_hash:
        failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "candidate workload identity does not match receipt workload"))
    if tuple(receipt.artifacts) != tuple(receipt.workload.artifacts):
        failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "receipt artifacts do not match workload artifacts"))
    if workload is not None:
        expected_workload = _coerce_workload(workload)
        if expected_workload.workload_hash != receipt.workload.workload_hash:
            failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "receipt workload does not match dispatch workload"))
    if candidate is not None:
        expected_candidate = _coerce_candidate(candidate, receipt.workload)
        if expected_candidate.candidate_id != receipt.candidate.candidate_id:
            failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "receipt candidate does not match expected candidate"))
    if policy is not None:
        expected_policy = _coerce_policy(policy)
        if expected_policy != receipt.policy:
            failures.append(_failure(FailureCode.INVALID_POLICY, "receipt policy does not match expected policy"))
    if runtime is not None:
        expected_runtime = _coerce_runtime(runtime)
        if expected_runtime.identity != receipt.runtime.identity:
            failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "receipt runtime does not match expected runtime"))

    if receipt.evaluator_bundle is not None:
        bundle_data = _thaw_json(receipt.evaluator_bundle)
        try:
            bundle_runtime = RuntimeIdentity.from_dict(bundle_data["runtime"])
            if bundle_data["candidate_id"] != receipt.candidate.candidate_id:
                failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "evaluator bundle candidate does not match receipt"))
            if bundle_data["workload_hash"] != receipt.workload.workload_hash:
                failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "evaluator bundle workload does not match receipt"))
            if bundle_runtime.identity != receipt.runtime.identity:
                failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "evaluator bundle runtime does not match receipt"))
        except (KeyError, ContractError) as exc:
            failures.append(_failure(FailureCode.IDENTITY_MISMATCH, f"evaluator bundle identity is malformed: {exc}"))

    try:
        recomputed = recompute_receipt_fields(
            receipt.raw_samples,
            receipt.policy,
            evaluator_bundle=None if receipt.evaluator_bundle is None else _thaw_json(receipt.evaluator_bundle),
        )
    except ContractError as exc:
        failures.append(Failure(exc.code, str(exc)))
        recomputed = {}
    else:
        for field in ("aggregates", "oracle", "compatibility", "metrics", "statistics"):
            if _thaw_json(getattr(receipt, field)) != recomputed.get(field):
                failures.append(_failure(FailureCode.IDENTITY_MISMATCH, f"receipt {field} was not recomputed from raw samples"))
        if receipt.evaluator_bundle is not None and not _raw_samples_match_evaluator_bundle(
            receipt.raw_samples, _thaw_json(receipt.evaluator_bundle)
        ):
            failures.append(_failure(FailureCode.IDENTITY_MISMATCH, "raw samples do not match retained evaluator records"))

    indices = [sample.index for sample in receipt.raw_samples]
    if receipt.evaluator_bundle is None:
        expected_indices = list(range(receipt.policy.measurement_runs))
        samples_complete = indices == expected_indices and len(indices) == receipt.policy.measurement_runs
    else:
        # Wave B: the real block count varies with sequential sampling
        # (policy.measurement_runs is only the starting/minimum count), so
        # the expected sample count comes from the independently recomputed
        # compatibility.block_count_used, not the policy field directly.
        block_count_used = recomputed.get("compatibility", {}).get("block_count_used") if recomputed else None
        block_count_in_range = (
            type(block_count_used) is int
            and receipt.policy.measurement_runs <= block_count_used <= receipt.policy.max_measurement_runs
        )
        if not block_count_in_range:
            expected_indices = []
            samples_complete = False
        else:
            expected_indices = list(range(block_count_used * 4))
            samples_complete = (
                indices == expected_indices
                and len(indices) == block_count_used * 4
                and bool(recomputed)
                and recomputed.get("compatibility", {}).get("compatible") is True
            )
    if not samples_complete:
        failures.append(
            _failure(
                FailureCode.INVALID_VALUE,
                "raw samples are incomplete or out of order",
                expected=expected_indices,
                observed=indices,
            )
        )
    if receipt.status not in {"complete", "failed"}:
        failures.append(_failure(FailureCode.INVALID_VALUE, "receipt status is not closed"))
    if receipt.status == "complete" and receipt.failure is not None:
        failures.append(_failure(FailureCode.INVALID_VALUE, "complete receipt carries a failure"))
    if receipt.status == "failed" and receipt.failure is None:
        failures.append(_failure(FailureCode.RUNTIME_FAILURE, "failed receipt has no failure classification"))
    if recomputed.get("oracle", {}).get("all_match") is False:
        failures.append(_failure(FailureCode.ORACLE_MISMATCH, "candidate output does not match the source oracle"))
    if recomputed.get("compatibility", {}).get("compatible") is False:
        failures.append(_failure(FailureCode.INVALID_POLICY, "raw sample count is incompatible with evaluation policy"))
    if receipt.status == "complete" and any(
        sample.duration_ns <= 0 or sample.baseline_duration_ns <= 0 for sample in receipt.raw_samples
    ):
        failures.append(_failure(FailureCode.RUNTIME_FAILURE, "complete receipt contains fabricated or missing zero timing"))

    artifacts_verified = False
    if artifact_root is not None:
        before_artifact_failures = len(failures)
        try:
            root_fd = _open_root_directory(artifact_root)
            os.close(root_fd)
        except ContractError as exc:
            failures.append(Failure(exc.code, str(exc)))
        for artifact in receipt.artifacts:
            try:
                verify_artifact(artifact_root, artifact)
            except ContractError as exc:
                failures.append(Failure(exc.code, str(exc)))
        artifacts_verified = len(failures) == before_artifact_failures

    attestation_value: str | None = None
    attested = False
    if type(attestation) is str:
        try:
            validate_sha256(attestation)
            attestation_value = attestation
        except ContractError:
            attestation_value = None
    elif type(attestation) in {bytes, bytearray}:
        try:
            decoded = bytes(attestation).decode("ascii")
            validate_sha256(decoded)
            attestation_value = decoded
        except (UnicodeDecodeError, ContractError):
            attestation_value = None
    if attestation_value is not None and type(attestation_key) in {bytes, bytearray}:
        try:
            expected_attestation = receipt_attestation(receipt, bytes(attestation_key))
            attested = hmac.compare_digest(attestation_value, expected_attestation)
        except ContractError:
            attested = False

    return ReceiptValidation(
        receipt=receipt,
        valid=not failures,
        complete=samples_complete and receipt.failure is None,
        local=receipt.to_dict().get("provenance") == "local_evaluation",
        failures=tuple(failures),
        recomputed=recomputed,
        attested=attested,
        attestation=attestation_value if attested else None,
        artifacts_verified=artifacts_verified,
        _proof=_ValidationProof(_VALIDATION_TOKEN, receipt.receipt_id, attestation_value if attested else None),
    )


def _fsync(fd: int) -> bool:
    try:
        os.fsync(fd)
        return True
    except OSError as exc:
        unsupported = {errno.EINVAL, errno.ENOTSUP, errno.EBADF, 95}
        if hasattr(errno, "EOPNOTSUPP"):
            unsupported.add(errno.EOPNOTSUPP)
        if exc.errno not in unsupported:
            raise
        return False


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or os.open not in getattr(os, "supports_dir_fd", ()):
        raise OSError(errno.ENOTSUP, "safe descriptor-relative directory access is unavailable")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_or_create_absolute_directory(path: str | os.PathLike[str]) -> int:
    raw = os.fspath(path)
    if not os.path.isabs(raw):
        raw = os.path.abspath(raw)
    input_path = Path(raw)
    expected_identity: tuple[int, int] | None = None
    try:
        input_stat = os.lstat(input_path)
        if stat.S_ISLNK(input_stat.st_mode):
            raise OSError(errno.ELOOP, "storage root must not be a symlink")
        expected_identity = (input_stat.st_dev, input_stat.st_ino)
        path_obj = input_path.resolve(strict=True)
    except FileNotFoundError:
        # Resolve existing parents (including platform aliases such as macOS
        # /var) while retaining the not-yet-created final component.
        path_obj = input_path.resolve(strict=False)
    flags = _directory_flags()
    descriptor = os.open(path_obj.anchor, flags)
    try:
        for component in path_obj.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        final_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(final_stat.st_mode):
            raise OSError(errno.ENOTDIR, "storage root is not a directory")
        if expected_identity is not None and (final_stat.st_dev, final_stat.st_ino) != expected_identity:
            raise OSError(errno.EAGAIN, "storage root changed while it was being opened")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(parent_fd: int, name: str, *, create: bool) -> int:
    flags = _directory_flags()
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        os.mkdir(name, 0o755, dir_fd=parent_fd)
        return os.open(name, flags, dir_fd=parent_fd)


def _write_exclusive_json(
    root: str | os.PathLike[str],
    kind: str,
    identity: str,
    payload: bytes,
    *,
    durability: list[bool] | None = None,
) -> Path:
    validate_sha256(identity)
    root_fd = _open_or_create_absolute_directory(root)
    kind_fd: int | None = None
    created = False
    filename = f"{identity}.json"
    try:
        kind_fd = _open_child_directory(root_fd, kind, create=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(filename, flags, 0o644, dir_fd=kind_fd)
        created = True
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            if durability is not None:
                durability[0] = durability[0] and _fsync(fd)
            else:
                _fsync(fd)
        finally:
            os.close(fd)
        if durability is not None:
            durability[0] = durability[0] and _fsync(kind_fd)
            durability[0] = durability[0] and _fsync(root_fd)
        else:
            _fsync(kind_fd)
            _fsync(root_fd)
        return Path(os.fspath(root)) / kind / filename
    except BaseException:
        if created and kind_fd is not None:
            try:
                os.unlink(filename, dir_fd=kind_fd)
                _fsync(kind_fd)
            except OSError:
                pass
        raise
    finally:
        if kind_fd is not None:
            os.close(kind_fd)
        os.close(root_fd)


def _read_existing(root: str | os.PathLike[str], kind: str, identity: str) -> bytes:
    validate_sha256(identity)
    if kind == "receipts":
        read_cap = MAX_STORED_RECEIPT_BYTES
    elif kind == "decisions":
        read_cap = MAX_STORED_DECISION_BYTES
    elif kind == "tuning_summaries":
        read_cap = MAX_STORED_TUNING_SUMMARY_BYTES
    else:
        raise ContractError("unknown content-addressed object kind", code=FailureCode.INVALID_VALUE)
    root_fd = _open_or_create_absolute_directory(root)
    kind_fd: int | None = None
    fd: int | None = None
    try:
        kind_fd = _open_child_directory(root_fd, kind, create=False)
        flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        fd = os.open(f"{identity}.json", flags, dir_fd=kind_fd)
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ContractError("stored object is not a regular file", code=FailureCode.ARTIFACT_NOT_REGULAR)
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = None
            payload = handle.read(read_cap + 1)
            if len(payload) > read_cap:
                raise ContractError(
                    f"stored {kind[:-1]} exceeds the {read_cap}-byte read cap",
                    code=FailureCode.IDENTITY_MISMATCH,
                )
            return payload
    except FileNotFoundError as exc:
        raise ContractError("content-addressed object is missing", code=FailureCode.ARTIFACT_MISSING) from exc
    except OSError as exc:
        raise ContractError(f"cannot read content-addressed object: {exc}", code=FailureCode.IDENTITY_MISMATCH) from exc
    finally:
        if fd is not None:
            os.close(fd)
        if kind_fd is not None:
            os.close(kind_fd)
        os.close(root_fd)


def _list_kind_ids(root: str | os.PathLike[str], kind: str) -> tuple[str, ...]:
    """Every valid content-addressed identity stored under one store kind.

    Fails open (empty tuple) on a kind directory that does not exist yet --
    listing a fresh store before anything of that kind was ever written is
    not an error. A stray or malformed filename is skipped rather than
    raised on: this is a read-only enumeration aid (Wave C's ``history``
    command), not itself a trust boundary -- every id it returns is still
    independently re-verified by :meth:`ContentAddressedStore.get_receipt`/
    ``get_tuning_summary`` before any of its content is trusted.
    """

    root_fd = _open_or_create_absolute_directory(root)
    kind_fd: int | None = None
    try:
        try:
            kind_fd = _open_child_directory(root_fd, kind, create=False)
        except FileNotFoundError:
            return ()
        identities: list[str] = []
        for name in os.listdir(kind_fd):
            if not name.endswith(".json"):
                continue
            candidate = name[: -len(".json")]
            try:
                identities.append(validate_sha256(candidate))
            except ContractError:
                continue
        return tuple(sorted(identities))
    finally:
        if kind_fd is not None:
            os.close(kind_fd)
        os.close(root_fd)


def _write_pointer(
    root: str | os.PathLike[str],
    name: str,
    value: str,
    *,
    durability: list[bool] | None = None,
) -> Path:
    _string(value, label="pointer value")
    root_fd = _open_or_create_absolute_directory(root)
    pointers_fd: int | None = None
    tmp_name = f".{name}.{uuid.uuid4().hex}.tmp"
    sync_error: OSError | None = None
    try:
        pointers_fd = _open_child_directory(root_fd, "pointers", create=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp_name, flags, 0o644, dir_fd=pointers_fd)
        try:
            payload = value.encode("ascii") + b"\n"
            os.write(fd, payload)
            try:
                if durability is not None:
                    durability[0] = durability[0] and _fsync(fd)
                else:
                    _fsync(fd)
            except OSError as exc:
                sync_error = exc
        finally:
            os.close(fd)
        os.replace(tmp_name, name, src_dir_fd=pointers_fd, dst_dir_fd=pointers_fd)
        try:
            if durability is not None:
                durability[0] = durability[0] and _fsync(pointers_fd)
                durability[0] = durability[0] and _fsync(root_fd)
            else:
                _fsync(pointers_fd)
                _fsync(root_fd)
        except OSError as exc:
            if sync_error is None:
                sync_error = exc
        if sync_error is not None:
            raise sync_error
        return Path(os.fspath(root)) / "pointers" / name
    finally:
        if pointers_fd is not None:
            try:
                os.unlink(tmp_name, dir_fd=pointers_fd)
            except FileNotFoundError:
                pass
            os.close(pointers_fd)
        os.close(root_fd)


def _read_pointer(root: str | os.PathLike[str], name: str) -> str:
    root_fd = _open_or_create_absolute_directory(root)
    pointers_fd: int | None = None
    fd: int | None = None
    try:
        pointers_fd = _open_child_directory(root_fd, "pointers", create=False)
        flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        fd = os.open(name, flags, dir_fd=pointers_fd)
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = None
            payload = handle.read(MAX_CURRENT_POINTER_BYTES + 1)
            if len(payload) > MAX_CURRENT_POINTER_BYTES:
                raise ContractError("current pointer is too large", code=FailureCode.IDENTITY_MISMATCH)
            if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
                raise ContractError("current pointer is not a canonical line", code=FailureCode.IDENTITY_MISMATCH)
            value = payload[:-1].decode("ascii")
            if not value:
                raise ContractError("current pointer is empty", code=FailureCode.IDENTITY_MISMATCH)
        if value != NATIVE_FALLBACK:
            validate_sha256(value)
        return value
    except FileNotFoundError as exc:
        raise ContractError("current pointer is missing", code=FailureCode.FALLBACK) from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError("current pointer is malformed", code=FailureCode.IDENTITY_MISMATCH) from exc
    finally:
        if fd is not None:
            os.close(fd)
        if pointers_fd is not None:
            os.close(pointers_fd)
        os.close(root_fd)


def _history_index_name(workload_hash: str, runtime_identity: str) -> str:
    """A stable, content-addressed index filename for one (workload, runtime) pair.

    Wave C (``auto-mlx tune``/``history``): tuning summaries are immutable,
    content-addressed objects with no identity that names the workload or
    runtime they came from, so a caller cannot enumerate "prior summaries
    for workload X on this machine" from the summary store alone (exactly
    like receipts -- see the module docstring).  This index is the one
    piece of *mutable* state Wave C introduces: an append-only, atomically
    replaced list of summary_ids, keyed by a hash of the (workload_hash,
    runtime_identity) pair it belongs to -- deliberately mirroring
    MetaSchedule's split between an immutable tuning record and a mutable
    workload-identity-keyed index into those records.
    """

    validate_sha256(workload_hash)
    validate_sha256(runtime_identity)
    return sha256_hex({"workload_hash": workload_hash, "runtime_identity": runtime_identity})


def _write_history_index(
    root: str | os.PathLike[str],
    index_name: str,
    summary_ids: Sequence[str],
    *,
    durability: list[bool] | None = None,
) -> Path:
    validate_sha256(index_name)
    for summary_id in summary_ids:
        validate_sha256(summary_id)
    payload = canonical_bytes(list(summary_ids))
    if len(payload) > MAX_STORED_HISTORY_INDEX_BYTES:
        raise ContractError("tuning history index exceeds its byte cap", code=FailureCode.IDENTITY_MISMATCH)
    root_fd = _open_or_create_absolute_directory(root)
    history_fd: int | None = None
    tmp_name = f".{index_name}.{uuid.uuid4().hex}.tmp"
    sync_error: OSError | None = None
    try:
        history_fd = _open_child_directory(root_fd, "tuning_history", create=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp_name, flags, 0o644, dir_fd=history_fd)
        try:
            os.write(fd, payload)
            try:
                if durability is not None:
                    durability[0] = durability[0] and _fsync(fd)
                else:
                    _fsync(fd)
            except OSError as exc:
                sync_error = exc
        finally:
            os.close(fd)
        os.replace(tmp_name, f"{index_name}.json", src_dir_fd=history_fd, dst_dir_fd=history_fd)
        try:
            if durability is not None:
                durability[0] = durability[0] and _fsync(history_fd)
                durability[0] = durability[0] and _fsync(root_fd)
            else:
                _fsync(history_fd)
                _fsync(root_fd)
        except OSError as exc:
            if sync_error is None:
                sync_error = exc
        if sync_error is not None:
            raise sync_error
        return Path(os.fspath(root)) / "tuning_history" / f"{index_name}.json"
    finally:
        if history_fd is not None:
            try:
                os.unlink(tmp_name, dir_fd=history_fd)
            except FileNotFoundError:
                pass
            os.close(history_fd)
        os.close(root_fd)


def _read_history_index(root: str | os.PathLike[str], index_name: str) -> tuple[str, ...]:
    validate_sha256(index_name)
    root_fd = _open_or_create_absolute_directory(root)
    history_fd: int | None = None
    fd: int | None = None
    try:
        history_fd = _open_child_directory(root_fd, "tuning_history", create=False)
        flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        fd = os.open(f"{index_name}.json", flags, dir_fd=history_fd)
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = None
            payload = handle.read(MAX_STORED_HISTORY_INDEX_BYTES + 1)
            if len(payload) > MAX_STORED_HISTORY_INDEX_BYTES:
                raise ContractError("tuning history index exceeds its byte cap", code=FailureCode.IDENTITY_MISMATCH)
        parsed = strict_json_loads(payload)
        if canonical_bytes(parsed) != payload:
            raise ContractError("stored tuning history index is not canonical", code=FailureCode.IDENTITY_MISMATCH)
        if type(parsed) is not list or any(type(item) is not str for item in parsed):
            raise ContractError("stored tuning history index must be an array of strings", code=FailureCode.WRONG_TYPE)
        for summary_id in parsed:
            validate_sha256(summary_id)
        return tuple(parsed)
    except FileNotFoundError:
        return ()
    finally:
        if fd is not None:
            os.close(fd)
        if history_fd is not None:
            os.close(history_fd)
        os.close(root_fd)


class ContentAddressedStore:
    """Filesystem store with immutable objects and atomic mutable pointers."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self._last_durable = True

    @property
    def last_durable(self) -> bool:
        """Whether the most recent write reached all supported fsync points."""

        return self._last_durable

    def _durability_error(self) -> ContractError:
        return ContractError(
            "filesystem durability could not be established",
            code=FailureCode.RUNTIME_FAILURE,
        )

    def put_receipt(self, receipt: Receipt | Mapping[str, Any], *, require_durable: bool = False) -> Path:
        value = receipt if isinstance(receipt, Receipt) else Receipt.from_dict(receipt)
        if value.receipt_id != sha256_hex(value._body_dict()):
            raise ContractError("cannot store receipt with a forged identity", code=FailureCode.IDENTITY_MISMATCH)
        durability = [True]
        result = _write_exclusive_json(
            self.root,
            "receipts",
            value.receipt_id,
            canonical_bytes(value.to_dict()),
            durability=durability,
        )
        self._last_durable = durability[0]
        if require_durable and not durability[0]:
            raise self._durability_error()
        return result

    store_receipt = put_receipt
    write_receipt = put_receipt

    def get_receipt(self, receipt_id: str) -> Receipt:
        payload = _read_existing(self.root, "receipts", receipt_id)
        parsed = strict_json_loads(payload)
        if canonical_bytes(parsed) != payload:
            raise ContractError("stored receipt is not canonical", code=FailureCode.IDENTITY_MISMATCH)
        receipt = Receipt.from_dict(parsed)
        if receipt.receipt_id != receipt_id:
            raise ContractError("stored receipt identity does not match path", code=FailureCode.IDENTITY_MISMATCH)
        return receipt

    read_receipt = get_receipt
    load_receipt = get_receipt

    def list_receipt_ids(self) -> tuple[str, ...]:
        """Every receipt_id stored under this store's ``receipts`` kind.

        Read-only enumeration for ``auto-mlx history`` (see
        ``auto_mlx.tuning.list_tuning_history``); callers still independently
        re-verify each entry through :meth:`get_receipt` before trusting it.
        """

        return _list_kind_ids(self.root, "receipts")

    def validate_stored_receipt(self, receipt_id: str, **kwargs: Any) -> ReceiptValidation:
        try:
            receipt = self.get_receipt(receipt_id)
        except ContractError as exc:
            return ReceiptValidation(
                None,
                False,
                False,
                False,
                (Failure(exc.code, str(exc)),),
                {},
                _proof=_ValidationProof(_VALIDATION_TOKEN, None, None),
            )
        return validate_receipt(receipt, **kwargs)

    def set_current_receipt(self, receipt_id: str, *, require_durable: bool = False) -> Path:
        validate_sha256(receipt_id)
        durability = [True]
        result = _write_pointer(self.root, "current_receipt", receipt_id, durability=durability)
        self._last_durable = durability[0]
        if require_durable and not durability[0]:
            raise self._durability_error()
        return result

    def current_receipt_id(self) -> str:
        return _read_pointer(self.root, "current_receipt")

    def put_decision(self, decision: ContractMapping | Mapping[str, Any], *, require_durable: bool = False) -> Path:
        data = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        # Import lazily to keep the receipt module usable by promotion without
        # a module-level cycle, while still enforcing the closed decision wire
        # shape at the storage boundary.
        from .promotion import PromotionDecision

        parsed_decision = PromotionDecision.from_dict(data)
        data = parsed_decision.to_dict()
        identity = data.get("decision_id")
        if type(identity) is not str:
            raise ContractError("decision must expose decision_id", code=FailureCode.WRONG_TYPE)
        body = dict(data)
        body.pop("decision_id", None)
        if identity != sha256_hex(body):
            raise ContractError("cannot store decision with a forged identity", code=FailureCode.IDENTITY_MISMATCH)
        durability = [True]
        try:
            result = _write_exclusive_json(
                self.root,
                "decisions",
                identity,
                canonical_bytes(data),
                durability=durability,
            )
        except FileExistsError:
            raise
        except OSError as exc:
            raise self._durability_error() from exc
        self._last_durable = durability[0]
        if require_durable and not durability[0]:
            raise self._durability_error()
        return result

    store_decision = put_decision
    write_decision = put_decision

    def get_decision(self, decision_id: str) -> dict[str, Any]:
        payload = _read_existing(self.root, "decisions", decision_id)
        parsed = strict_json_loads(payload)
        if canonical_bytes(parsed) != payload:
            raise ContractError("stored decision is not canonical", code=FailureCode.IDENTITY_MISMATCH)
        if not isinstance(parsed, dict) or parsed.get("decision_id") != decision_id:
            raise ContractError("stored decision identity does not match path", code=FailureCode.IDENTITY_MISMATCH)
        body = dict(parsed)
        body.pop("decision_id", None)
        if sha256_hex(body) != decision_id:
            raise ContractError("stored decision hash does not match body", code=FailureCode.IDENTITY_MISMATCH)
        from .promotion import PromotionDecision

        PromotionDecision.from_dict(parsed)
        return parsed

    read_decision = get_decision
    load_decision = get_decision

    def set_current_decision(self, decision_id: str, *, require_durable: bool = False) -> Path:
        if decision_id != NATIVE_FALLBACK:
            validate_sha256(decision_id)
        durability = [True]
        result = _write_pointer(self.root, "current_decision", decision_id, durability=durability)
        self._last_durable = durability[0]
        if require_durable and not durability[0]:
            raise self._durability_error()
        return result

    set_current = set_current_decision

    def current_decision_id(self) -> str:
        return _read_pointer(self.root, "current_decision")

    current_id = current_decision_id

    def put_tuning_summary(self, summary: ContractMapping | Mapping[str, Any], *, require_durable: bool = False) -> Path:
        """Store a Wave C tuning summary content-addressed, alongside receipts.

        Imports :class:`auto_mlx.tune.TuningSummary` lazily -- the same
        deliberate late-import used by ``put_decision``/``get_decision`` for
        :class:`auto_mlx.promotion.PromotionDecision`, so this module never
        gains a module-level dependency on the tuning layer built on top of
        it.
        """

        data = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
        from .tuning import TuningSummary

        parsed_summary = TuningSummary.from_dict(data)
        data = parsed_summary.to_dict()
        identity = data.get("summary_id")
        if type(identity) is not str:
            raise ContractError("tuning summary must expose summary_id", code=FailureCode.WRONG_TYPE)
        body = dict(data)
        body.pop("summary_id", None)
        if identity != sha256_hex(body):
            raise ContractError("cannot store tuning summary with a forged identity", code=FailureCode.IDENTITY_MISMATCH)
        durability = [True]
        try:
            result = _write_exclusive_json(
                self.root,
                "tuning_summaries",
                identity,
                canonical_bytes(data),
                durability=durability,
            )
        except FileExistsError:
            raise
        except OSError as exc:
            raise self._durability_error() from exc
        self._last_durable = durability[0]
        if require_durable and not durability[0]:
            raise self._durability_error()
        return result

    store_tuning_summary = put_tuning_summary
    write_tuning_summary = put_tuning_summary

    def get_tuning_summary(self, summary_id: str) -> dict[str, Any]:
        payload = _read_existing(self.root, "tuning_summaries", summary_id)
        parsed = strict_json_loads(payload)
        if canonical_bytes(parsed) != payload:
            raise ContractError("stored tuning summary is not canonical", code=FailureCode.IDENTITY_MISMATCH)
        if not isinstance(parsed, dict) or parsed.get("summary_id") != summary_id:
            raise ContractError("stored tuning summary identity does not match path", code=FailureCode.IDENTITY_MISMATCH)
        body = dict(parsed)
        body.pop("summary_id", None)
        if sha256_hex(body) != summary_id:
            raise ContractError("stored tuning summary hash does not match body", code=FailureCode.IDENTITY_MISMATCH)
        from .tuning import TuningSummary

        TuningSummary.from_dict(parsed)
        return parsed

    read_tuning_summary = get_tuning_summary
    load_tuning_summary = get_tuning_summary

    def list_tuning_summary_ids(self) -> tuple[str, ...]:
        """Every summary_id stored under this store's ``tuning_summaries`` kind.

        Read-only enumeration for ``auto-mlx history``; callers still
        independently re-verify each entry through :meth:`get_tuning_summary`
        before trusting it.
        """

        return _list_kind_ids(self.root, "tuning_summaries")

    def append_tuning_history(self, workload_hash: str, runtime_identity: str, summary_id: str, *, require_durable: bool = False) -> Path:
        """Append ``summary_id`` to the (workload, runtime) history index.

        Read-modify-write over the small atomic-replace index (see
        ``_write_history_index``/``_read_history_index`` above) -- adequate
        for this store's local, single-operator scope (the same scope every
        other mutable pointer in this module already assumes). A
        already-present summary_id is not duplicated.
        """

        validate_sha256(summary_id)
        index_name = _history_index_name(workload_hash, runtime_identity)
        existing = _read_history_index(self.root, index_name)
        if summary_id in existing:
            return Path(os.fspath(self.root)) / "tuning_history" / f"{index_name}.json"
        durability = [True]
        result = _write_history_index(self.root, index_name, existing + (summary_id,), durability=durability)
        self._last_durable = durability[0]
        if require_durable and not durability[0]:
            raise self._durability_error()
        return result

    def list_tuning_history(self, workload_hash: str, runtime_identity: str) -> tuple[str, ...]:
        index_name = _history_index_name(workload_hash, runtime_identity)
        return _read_history_index(self.root, index_name)


def receipt_identity(value: Receipt | Mapping[str, Any]) -> str:
    receipt = value if isinstance(value, Receipt) else Receipt.from_dict(value)
    return sha256_hex(receipt._body_dict())


def write_receipt(store: ContentAddressedStore, receipt: Receipt | Mapping[str, Any]) -> Path:
    return store.put_receipt(receipt)


def read_receipt(store: ContentAddressedStore, receipt_id: str) -> Receipt:
    return store.get_receipt(receipt_id)


ReceiptStore = ContentAddressedStore


__all__: Final = [
    "adapt_observation_bundle",
    "attest_receipt",
    "CLAIMS_WITHHELD",
    "ContentAddressedStore",
    "ContractMapping",
    "MAX_STORED_HISTORY_INDEX_BYTES",
    "MAX_STORED_TUNING_SUMMARY_BYTES",
    "NATIVE_FALLBACK",
    "RawSample",
    "RECEIPT_SCHEMA",
    "Receipt",
    "ReceiptStore",
    "ReceiptValidation",
    "receipt_attestation",
    "receipt_from_observation_bundle",
    "recompute_receipt_fields",
    "receipt_identity",
    "read_receipt",
    "validate_receipt",
    "write_receipt",
]
