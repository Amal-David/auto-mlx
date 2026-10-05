"""Offline serving identity/comparability contracts, not benchmark attestations.

These original contracts encode lessons from the pinned engine survey. They
never import an engine, activate a candidate, or implement a KV cache.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import re
from typing import Any

from .canonical import sha256_hex, validate_json_value
from .errors import ContractError, FailureCode
from .paths import validate_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message, code=FailureCode.INVALID_VALUE)


def _closed(value: Any, expected: set[str], label: str) -> dict:
    validate_json_value(value)
    _require(type(value) is dict and set(value) == expected, f"{label} must contain exactly its declared fields")
    return value


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """Identity prerequisite for future state reuse; a hash is not authorization."""
    model_sha256: str
    tokenizer_sha256: str
    processor_sha256: str
    template_sha256: str
    adapter_sha256: str
    draft_sha256: str
    cache_layout_sha256: str
    tenant_namespace_sha256: str
    ordered_media_sha256: str
    cache_abi: str

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name != "cache_abi":
                validate_sha256(value)
            else:
                _require(type(value) is str and re.fullmatch(r"[a-zA-Z0-9._/-]{1,128}", value) is not None, "invalid cache ABI identity")

    def to_dict(self) -> dict[str, str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def identity(self) -> str:
        return sha256_hex({"schema": "auto-mlx-cache-namespace-v1", **self.to_dict()})

    @classmethod
    def from_dict(cls, value: Any) -> "CacheNamespace":
        return cls(**_closed(value, {field.name for field in fields(cls)}, "cache namespace"))


METRICS = {
    "request_first_content_ns": "ns",
    "request_end_to_end_ns": "ns",
    "physical_prefill_tokens_per_second": "tokens_per_second",
    "logical_prefill_tokens_per_second": "tokens_per_second",
    "decode_tokens_per_second": "tokens_per_second",
    "aggregate_output_tokens_per_second": "tokens_per_second",
    "process_peak_rss_bytes": "bytes",
    "metal_allocated_bytes": "bytes",
    "first_audio_ns": "ns",
    "audio_realtime_factor_bps": "basis_points",
}


@dataclass(frozen=True, slots=True)
class ServingComparisonContext:
    """A declared comparison cell, independent of engine and measurement numbers.

    Different engine implementations are allowed. The model, workload, quality,
    cache regime, hardware and metric meaning cannot silently change with them.
    Even a passing comparison does not attest the declarations or promote a win.
    """
    engine: str
    engine_revision: str
    model_sha256: str
    tokenizer_processor_sha256: str
    template_adapter_sha256: str
    workload_sha256: str
    sampling_policy_sha256: str
    quality_contract_sha256: str
    hardware_sha256: str
    measurement_protocol_sha256: str
    cache_state: str
    cache_precision: str
    model_load_state: str
    concurrency: int
    input_tokens: int
    max_output_tokens: int
    metric: str
    unit: str
    quality_passed: bool

    def __post_init__(self) -> None:
        validate_json_value(self.to_dict())
        _require(type(self.engine) is str and 0 < len(self.engine) <= 128, "engine must be a bounded name")
        _require(type(self.engine_revision) is str and re.fullmatch(r"[0-9a-f]{40}", self.engine_revision) is not None, "engine revision must be an immutable commit")
        for field in fields(self):
            if field.name.endswith("_sha256"):
                validate_sha256(getattr(self, field.name))
        for key, low, high in (("concurrency", 1, 4096), ("input_tokens", 1, 16777216), ("max_output_tokens", 1, 1048576)):
            value = getattr(self, key)
            _require(type(value) is int and low <= value <= high, f"{key} is outside its integer bound")
        _require(type(self.cache_state) is str and self.cache_state in {"cold", "warm", "partial"}, "invalid cache regime")
        _require(type(self.model_load_state) is str and self.model_load_state in {"cold", "resident"}, "invalid model load state")
        _require(type(self.cache_precision) is str and 0 < len(self.cache_precision) <= 128, "cache precision/layout must be explicit")
        _require(type(self.metric) is str and self.metric in METRICS, "metric needs explicit semantics")
        _require(self.unit == METRICS[self.metric], "metric unit does not match its semantics")
        _require(type(self.quality_passed) is bool, "quality_passed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, value: Any) -> "ServingComparisonContext":
        return cls(**_closed(value, {field.name for field in fields(cls)}, "serving comparison context"))


def compare_contexts(baseline: ServingComparisonContext, candidate: ServingComparisonContext) -> dict[str, Any]:
    _require(type(baseline) is ServingComparisonContext and type(candidate) is ServingComparisonContext, "typed comparison contexts are required")
    differences = [field.name for field in fields(baseline) if field.name not in {"engine", "engine_revision", "quality_passed"} and getattr(baseline, field.name) != getattr(candidate, field.name)]
    blockers = [f"comparison field differs: {name}" for name in differences]
    if not baseline.quality_passed or not candidate.quality_passed:
        blockers.append("both sides must declare a passed shared quality contract")
    return {"schema_version": 1, "comparable": not blockers, "blockers": blockers, "baseline_engine": baseline.engine, "candidate_engine": candidate.engine, "scope": "declared-context-consistency-only-not-measurement-attestation", "promotion_allowed": False, "speedup": None}


def native_capabilities(device: str) -> dict[str, Any]:
    _require(type(device) is str and device in {"cpu", "gpu", "unknown"}, "invalid device")
    return {"runtime": "mlx-lm-native-preview", "configured_device": device, "device_execution_verified_by_this_report": False, "scope": "single-user-local", "compute_scheduling": "single-request-no-wait", "max_compute_batch_size": 1, "continuous_batching": False, "prefix_cache": "none", "ssd_cache": False, "weight_offload": False, "speculative_decoding": False, "endpoints": ["/v1/completions"], "sampling": "greedy-temperature-zero", "tools": False, "multimodal": False}
