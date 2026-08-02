"""Strict, deterministic JSON and SHA-256 identity helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .errors import CanonicalJSONError, DuplicateKeyError, FailureCode


def _reject_constant(value: str) -> None:
    raise CanonicalJSONError(
        f"non-finite JSON number {value!r} is not allowed",
        code=FailureCode.NON_FINITE_NUMBER,
    )


def _reject_float(value: str) -> None:
    raise CanonicalJSONError(
        f"JSON floating-point number {value!r} is not allowed",
        code=FailureCode.FLOAT_NOT_ALLOWED,
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate keys, floats, and non-finite values."""

    if not isinstance(value, (str, bytes, bytearray)):
        raise CanonicalJSONError("JSON input must be text or UTF-8 bytes")
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except CanonicalJSONError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CanonicalJSONError(f"invalid JSON: {exc}") from exc


def _validate_json_value(value: Any, *, path: str = "$", allow_tuple: bool = False) -> None:
    value_type = type(value)
    if value is None or value_type is str or value_type is bool or value_type is int:
        return
    if value_type is float:
        raise CanonicalJSONError(
            f"floating-point value at {path} is not allowed",
            code=FailureCode.FLOAT_NOT_ALLOWED,
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalJSONError(f"object key at {path} is not a string")
            _validate_json_value(item, path=f"{path}.{key}", allow_tuple=allow_tuple)
        return
    if isinstance(value, (list, tuple) if allow_tuple else list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]", allow_tuple=allow_tuple)
        return
    raise CanonicalJSONError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return compact sorted-key JSON encoded with UTF-8-compatible text."""

    _validate_json_value(value, allow_tuple=False)
    value = _json_ready(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CanonicalJSONError(f"cannot canonicalize JSON: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    """Return the canonical JSON representation as UTF-8 bytes."""

    try:
        return canonical_json(value).encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise CanonicalJSONError("JSON contains an unpaired UTF-16 surrogate") from exc


def sha256_hex(value: Any) -> str:
    """Hash a JSON value's canonical UTF-8 representation."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


# Short aliases make the contract helpers convenient without changing semantics.
loads = strict_json_loads
parse_json = strict_json_loads
canonicalize = canonical_json
canonical_json_bytes = canonical_bytes
identity = sha256_hex
sha256_identity = sha256_hex
digest = sha256_hex
