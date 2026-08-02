"""Stable error and failure-code vocabulary for the Auto MLX lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any


class FailureCode(str, Enum):
    """Machine-readable reasons that a contract or later lane can fail."""

    INVALID_JSON = "invalid_json"
    DUPLICATE_KEY = "duplicate_key"
    UNKNOWN_FIELD = "unknown_field"
    FLOAT_NOT_ALLOWED = "float_not_allowed"
    NON_FINITE_NUMBER = "non_finite_number"
    WRONG_TYPE = "wrong_type"
    INVALID_VALUE = "invalid_value"
    UNSAFE_PATH = "unsafe_path"
    INVALID_DIGEST = "invalid_digest"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_SYMLINK = "artifact_symlink"
    ARTIFACT_NOT_REGULAR = "artifact_not_regular"
    ARTIFACT_SIZE_MISMATCH = "artifact_size_mismatch"
    ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"
    INVALID_KNOB = "invalid_knob"
    CONFIG_MISMATCH = "config_mismatch"
    INVALID_POLICY = "invalid_policy"
    INVALID_RUNTIME = "invalid_runtime"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    OUTPUT_LIMIT = "output_limit"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    RUNTIME_FAILURE = "runtime_failure"
    ORACLE_MISMATCH = "oracle_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    PROMOTION_REJECTED = "promotion_rejected"
    FALLBACK = "fallback"
    INPUT_TOO_LARGE = "input_too_large"


def _freeze_detail(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_detail(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_detail(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_detail(item) for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _thaw_detail(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_detail(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_detail(item) for item in value]
    if isinstance(value, frozenset):
        return [_thaw_detail(item) for item in sorted(value, key=repr)]
    return value


@dataclass(frozen=True, slots=True)
class Failure:
    """A serializable failure value suitable for runner and promotion records."""

    code: FailureCode
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, FailureCode):
            raise TypeError("code must be a FailureCode")
        if type(self.message) is not str or not self.message:
            raise ValueError("message must be a non-empty string")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(self, "details", _freeze_detail(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": _thaw_detail(self.details),
        }


class AutoMLXError(Exception):
    """Base exception carrying a stable failure code."""

    code = FailureCode.INVALID_VALUE

    def __init__(self, message: str, *, code: FailureCode | None = None) -> None:
        super().__init__(message)
        self.code = code or type(self).code
        self.message = message

    def as_failure(self) -> Failure:
        return Failure(self.code, self.message)


class ContractError(AutoMLXError, ValueError):
    """Raised when untrusted input violates a G0 contract."""


class CanonicalJSONError(ContractError):
    code = FailureCode.INVALID_JSON


class DuplicateKeyError(CanonicalJSONError):
    code = FailureCode.DUPLICATE_KEY


class UnknownFieldError(ContractError):
    code = FailureCode.UNKNOWN_FIELD


class UnsafePathError(ContractError):
    code = FailureCode.UNSAFE_PATH


class ArtifactIntegrityError(ContractError):
    code = FailureCode.ARTIFACT_DIGEST_MISMATCH


# A short neutral name is useful to later runner/promotion code without making
# callers depend on the concrete base-class spelling.
Error = AutoMLXError
