"""Evaluator-owned exact-output correctness oracle.

The expected bytes are frozen when the oracle is constructed.  Candidate
proposals do not contain an oracle, threshold, or acceptance rule.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Any

from .contracts import Artifact
from .errors import ArtifactIntegrityError, ContractError, Failure, FailureCode
from .paths import _open_verified_file, validate_sha256


DEFAULT_MAX_ORACLE_ARTIFACT_BYTES: Final = 64 * 1024 * 1024


def _validate_max_bytes(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ContractError(
            "oracle max_bytes must be a positive integer",
            code=FailureCode.INVALID_POLICY if type(value) is int else FailureCode.WRONG_TYPE,
        )
    return value


@dataclass(frozen=True, slots=True)
class OracleResult:
    """One exact comparison, retaining only digest/size metadata for output."""

    matched: bool
    expected_digest: str
    actual_digest: str
    expected_size: int
    actual_size: int
    failure: Failure | None = None

    def __post_init__(self) -> None:
        if type(self.matched) is not bool:
            raise ContractError("oracle matched must be a boolean", code=FailureCode.WRONG_TYPE)
        validate_sha256(self.expected_digest)
        validate_sha256(self.actual_digest)
        for name in ("expected_size", "actual_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContractError(f"{name} must be a non-negative integer", code=FailureCode.WRONG_TYPE)
        if self.failure is not None and not isinstance(self.failure, Failure):
            raise ContractError("oracle failure must be a Failure or null", code=FailureCode.WRONG_TYPE)
        if self.matched and self.failure is not None:
            raise ContractError("a matched oracle result cannot carry failure metadata", code=FailureCode.ORACLE_MISMATCH)
        if not self.matched and self.failure is None:
            raise ContractError("a mismatched oracle result must carry failure metadata", code=FailureCode.ORACLE_MISMATCH)

    @property
    def passed(self) -> bool:
        return self.matched

    @property
    def exact_match(self) -> bool:
        return self.matched

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "expected_size": self.expected_size,
            "actual_size": self.actual_size,
            "failure": self.failure.to_dict() if self.failure else None,
        }


@dataclass(frozen=True, slots=True, init=False)
class ExactOutputOracle:
    """Immutable byte-for-byte oracle owned by the evaluator."""

    expected: bytes
    expected_digest: str
    label: str
    max_bytes: int

    def __init__(
        self,
        expected: bytes | bytearray,
        *,
        label: str = "exact-output",
        expected_digest: str | None = None,
        max_bytes: int = DEFAULT_MAX_ORACLE_ARTIFACT_BYTES,
    ) -> None:
        if type(expected) not in {bytes, bytearray}:
            raise ContractError("oracle expected output must be bytes", code=FailureCode.WRONG_TYPE)
        limit = _validate_max_bytes(max_bytes)
        if len(expected) > limit:
            raise ContractError(
                "oracle expected output exceeds the configured byte limit",
                code=FailureCode.OUTPUT_LIMIT,
            )
        frozen = bytes(expected)
        actual_digest = hashlib.sha256(frozen).hexdigest()
        if expected_digest is not None:
            validate_sha256(expected_digest)
            if expected_digest != actual_digest:
                raise ContractError("oracle expected digest does not match expected bytes", code=FailureCode.IDENTITY_MISMATCH)
        if type(label) is not str or not label:
            raise ContractError("oracle label must be a non-empty string", code=FailureCode.WRONG_TYPE)
        object.__setattr__(self, "expected", frozen)
        object.__setattr__(self, "expected_digest", actual_digest)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "max_bytes", limit)

    @classmethod
    def from_bytes(
        cls,
        expected: bytes | bytearray,
        *,
        label: str = "exact-output",
        max_bytes: int = DEFAULT_MAX_ORACLE_ARTIFACT_BYTES,
    ) -> "ExactOutputOracle":
        return cls(expected, label=label, max_bytes=max_bytes)

    @classmethod
    def from_artifact(
        cls,
        root: str | Path,
        artifact: Artifact,
        *,
        label: str = "exact-output",
        max_bytes: int = DEFAULT_MAX_ORACLE_ARTIFACT_BYTES,
    ) -> "ExactOutputOracle":
        limit = _validate_max_bytes(max_bytes)
        if not isinstance(artifact, Artifact):
            raise ContractError("oracle artifact must be an Artifact", code=FailureCode.WRONG_TYPE)
        if artifact.size_bytes > limit:
            raise ContractError(
                "oracle artifact exceeds the configured byte limit",
                code=FailureCode.OUTPUT_LIMIT,
            )
        descriptor = _open_verified_file(str(root), artifact.path)
        expected = bytearray()
        digest = hashlib.sha256()
        size = 0
        try:
            remaining = artifact.size_bytes
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise ArtifactIntegrityError(
                        "oracle expected artifact ended before its expected size",
                        code=FailureCode.ARTIFACT_SIZE_MISMATCH,
                    )
                expected.extend(chunk)
                digest.update(chunk)
                size += len(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ArtifactIntegrityError(
                    "oracle expected artifact exceeded its expected size",
                    code=FailureCode.ARTIFACT_SIZE_MISMATCH,
                )
        except OSError as exc:
            raise ContractError("oracle expected artifact could not be read", code=FailureCode.ARTIFACT_MISSING) from exc
        finally:
            os.close(descriptor)
        if size != artifact.size_bytes:
            raise ArtifactIntegrityError("oracle artifact size changed while being read", code=FailureCode.ARTIFACT_SIZE_MISMATCH)
        if digest.hexdigest() != artifact.sha256:
            raise ArtifactIntegrityError("oracle artifact changed while being read", code=FailureCode.IDENTITY_MISMATCH)
        return cls(bytes(expected), label=label, expected_digest=artifact.sha256, max_bytes=limit)

    def evaluate(self, actual: bytes | bytearray) -> OracleResult:
        if type(actual) not in {bytes, bytearray}:
            raise ContractError("oracle actual output must be bytes", code=FailureCode.WRONG_TYPE)
        if len(actual) > self.max_bytes:
            raise ContractError(
                "oracle actual output exceeds the configured byte limit",
                code=FailureCode.OUTPUT_LIMIT,
            )
        actual_bytes = bytes(actual)
        actual_digest = hashlib.sha256(actual_bytes).hexdigest()
        matched = (
            len(actual_bytes) == len(self.expected)
            and actual_digest == self.expected_digest
            and actual_bytes == self.expected
        )
        failure = None
        if not matched:
            failure = Failure(
                FailureCode.ORACLE_MISMATCH,
                "runner output did not exactly match the evaluator oracle",
                {
                    "label": self.label,
                    "expected_digest": self.expected_digest,
                    "actual_digest": actual_digest,
                    "expected_size": len(self.expected),
                    "actual_size": len(actual_bytes),
                },
            )
        return OracleResult(
            matched=matched,
            expected_digest=self.expected_digest,
            actual_digest=actual_digest,
            expected_size=len(self.expected),
            actual_size=len(actual_bytes),
            failure=failure,
        )

    compare = evaluate
    check = evaluate


__all__: Final = ["DEFAULT_MAX_ORACLE_ARTIFACT_BYTES", "ExactOutputOracle", "OracleResult"]
