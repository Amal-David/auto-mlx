"""Stable error and failure-code vocabulary for the Auto MLX lanes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any, Final


class FailureCode(str, Enum):
    """Machine-readable reasons that a contract or later lane can fail."""

    INVALID_JSON = "invalid_json"
    DUPLICATE_KEY = "duplicate_key"
    UNKNOWN_FIELD = "unknown_field"
    INVALID_UNICODE = "invalid_unicode"
    JSON_TOO_DEEP = "json_too_deep"
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
    ARTIFACT_ACCESS = "artifact_access"
    ARTIFACT_IO_ERROR = "artifact_io_error"
    ARTIFACT_SECURITY_UNAVAILABLE = "artifact_security_unavailable"
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
    KEY_MATERIAL_MISSING = "key_material_missing"
    KEY_MATERIAL_INVALID = "key_material_invalid"
    SUPERVISOR_REFUSED = "supervisor_refused"
    STORE_CONFIG_INVALID = "store_config_invalid"


MAX_JSON_DEPTH: Final = 64


def _contains_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def _validate_message(value: Any) -> str:
    """Validate an error message without routing failures through AutoMLXError."""

    if type(value) is not str:
        raise TypeError("message must be a string")
    if not value:
        raise ValueError("message must be a non-empty string")
    if _contains_surrogate(value):
        raise ValueError("message must not contain unpaired surrogates")
    return value


def _freeze_detail(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    if depth > MAX_JSON_DEPTH:
        raise TypeError(f"failure details exceed the maximum JSON nesting depth at {path}")
    value_type = type(value)
    if value is None or value_type is bool or value_type is int:
        return value
    if value_type is str:
        if _contains_surrogate(value):
            raise TypeError(f"failure details contain an unpaired surrogate at {path}")
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise TypeError(f"failure details contain a non-finite JSON number at {path}")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"failure detail object keys must be strings at {path}")
            if _contains_surrogate(key):
                raise TypeError(f"failure details contain an unpaired surrogate in an object key at {path}")
            frozen[key] = _freeze_detail(item, path=f"{path}.{key}", depth=depth + 1)
        return MappingProxyType(frozen)
    if value_type in {list, tuple}:
        return tuple(
            _freeze_detail(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise TypeError(f"failure details contain a non-JSON value at {path}: {type(value).__name__}")


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
        _validate_message(self.message)
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        frozen = _freeze_detail(self.details)
        try:
            json.dumps(_thaw_detail(frozen), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
            raise TypeError("details must be JSON-serializable contract data") from exc
        object.__setattr__(self, "details", frozen)

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
        validated_message = _validate_message(message)
        selected_code = type(self).code if code is None else code
        if not isinstance(selected_code, FailureCode):
            raise TypeError("code must be a FailureCode")
        super().__init__(validated_message)
        self.code = selected_code
        self.message = validated_message

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


class KeyMaterialError(AutoMLXError):
    """Raised when local attestation-key generation, storage, or loading fails closed.

    Never carries or logs the key itself -- only ever a description of what
    permission, symlink, or size check failed.  See ``auto_mlx.keys``.
    """

    code = FailureCode.KEY_MATERIAL_INVALID


class SupervisorRefusalError(AutoMLXError):
    """Raised when the local supervisor declines to attest a receipt.

    See ``auto_mlx.supervisor.attest_receipt`` -- this is the ONLY code path
    that mints a receipt attestation, and it raises this (never a silent
    False) on any independent-recompute, identity, or evidence-chain
    mismatch.
    """

    code = FailureCode.SUPERVISOR_REFUSED


class StoreConfigError(AutoMLXError):
    """Raised when the resolved receipt-store root and key directory conflict.

    See ``auto_mlx.store_config`` -- the store root and the attestation key
    directory must never nest inside one another.
    """

    code = FailureCode.STORE_CONFIG_INVALID


# A short neutral name is useful to later runner/promotion code without making
# callers depend on the concrete base-class spelling.
Error = AutoMLXError
