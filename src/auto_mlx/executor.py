"""Evaluator-owned, bounded subprocess execution for the G0 lane.

There is intentionally no MLX dependency here.  A candidate contributes only
typed configuration data.  Process creation is delegated to an evaluator-owned
``IsolationProvider``; process groups are cleanup machinery, never sandbox
evidence.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .canonical import canonical_bytes, sha256_hex, strict_json_loads
from .contracts import Artifact, CandidateProposal
from .errors import AutoMLXError, ContractError, Failure, FailureCode
from .paths import _open_verified_file, validate_sha256


_ARTIFACT_DIRECTORY = ".auto_mlx_artifacts"
_CONFIG_FILE = ".auto_mlx_candidate.json"
_REQUIRED_ISOLATION = frozenset({"network_denial", "descendant_containment"})
_MAX_REGISTERED_RUNNER_BYTES = 1 << 30
_INTERNAL_ENVIRONMENT = frozenset(
    {
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "XDG_CACHE_HOME",
        "AUTO_MLX_CONFIG_PATH",
        "AUTO_MLX_ARTIFACT_ROOT",
    }
)


def _non_empty_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"{label} must be a non-empty string", code=FailureCode.WRONG_TYPE)
    return value


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite positive number", code=FailureCode.INVALID_POLICY)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ContractError(f"{label} must be a finite positive number", code=FailureCode.INVALID_POLICY)
    return number


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ContractError(f"{label} must be a positive integer", code=FailureCode.INVALID_POLICY)
    return value


class _HandshakeTimeout(RuntimeError):
    """A bounded provider/authority call did not return in time."""


def _call_bounded(operation: Any, timeout_seconds: float) -> Any:
    """Run a handshake in a daemon thread when the call cannot be cancelled.

    Python cannot safely interrupt arbitrary provider code.  A daemon thread
    still lets the evaluator fail closed at a bounded time; callers must
    clean up any process handle they already own after an authority timeout.
    """

    result: list[Any] = []
    error: list[BaseException] = []

    def invoke() -> None:
        try:
            result.append(operation())
        except BaseException as exc:  # preserve unexpected authority failures
            error.append(exc)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise _HandshakeTimeout("bounded evaluator handshake timed out")
    if error:
        failure = error[0]
        if isinstance(failure, Exception):
            raise failure
        raise RuntimeError(f"evaluator handshake raised {type(failure).__name__}")
    if not result:
        raise RuntimeError("evaluator handshake returned no result")
    return result[0]


_LOCAL_SANDBOX_REQUIRED_BINARY: Final = "sandbox-exec"


def local_sandbox_primitives_available() -> bool:
    """Report whether this host has the primitives a local sandbox needs.

    This is a host-capability probe only: macOS, the ``sandbox-exec``
    (Seatbelt) binary on PATH, and the descriptor-relative ``O_NOFOLLOW``/
    ``dir_fd`` support the artifact layer already requires.  It says nothing
    about whether any specific :class:`IsolationProvider` is trustworthy;
    ``execute_plan`` still requires an explicit, evaluator-constructed
    provider and authority before it will run anything for real.  Checked at
    call time (never cached) so tests can simulate an unsupported host by
    adjusting ``PATH``.
    """

    if sys.platform != "darwin":
        return False
    if shutil.which(_LOCAL_SANDBOX_REQUIRED_BINARY) is None:
        return False
    if not hasattr(os, "O_NOFOLLOW") or getattr(os, "O_DIRECTORY", None) is None:
        return False
    if os.open not in getattr(os, "supports_dir_fd", ()):
        return False
    return True


def _freeze_environment(value: Mapping[str, str]) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise ContractError("extra_environment must be a mapping", code=FailureCode.WRONG_TYPE)
    frozen: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or not key or not key.replace("_", "A").isalnum() or not key[0].isalpha():
            raise ContractError("extra_environment keys must be environment variable names", code=FailureCode.INVALID_POLICY)
        if not key.startswith("AUTO_MLX_"):
            raise ContractError("extra_environment is restricted to evaluator AUTO_MLX_ keys", code=FailureCode.INVALID_POLICY)
        if key in _INTERNAL_ENVIRONMENT:
            raise ContractError(f"extra_environment cannot override {key}", code=FailureCode.INVALID_POLICY)
        if type(item) is not str:
            raise ContractError("extra_environment values must be strings", code=FailureCode.WRONG_TYPE)
        frozen[key] = item
    return MappingProxyType(frozen)


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    EXIT_FAILURE = "exit_failure"
    CRASH = "crash"
    TIMEOUT = "timeout"
    OUTPUT_FAILURE = "output_failure"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    START_FAILURE = "start_failure"
    ARTIFACT_FAILURE = "artifact_failure"


class CleanupMode(str, Enum):
    NOT_NEEDED = "not_needed"
    BEST_EFFORT_PROCESS_GROUP = "best_effort_process_group"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CleanupObservation:
    """Cleanup metadata; it is never presented as descendant-containment proof."""

    attempted: bool
    mode: CleanupMode
    verified: bool = False

    def __post_init__(self) -> None:
        if type(self.attempted) is not bool or type(self.verified) is not bool:
            raise ContractError("cleanup flags must be booleans", code=FailureCode.WRONG_TYPE)
        if not isinstance(self.mode, CleanupMode):
            raise ContractError("cleanup mode must be a CleanupMode", code=FailureCode.WRONG_TYPE)
        if self.verified:
            raise ContractError("process cleanup cannot be treated as verified containment", code=FailureCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True)
class IsolationClaim:
    """Untrusted provider metadata; it is not evidence of enforcement."""

    provider_id: str
    provider_identity: str
    requirements: frozenset[str]
    attestation_digest: str

    def __post_init__(self) -> None:
        _non_empty_string(self.provider_id, label="provider_id")
        validate_sha256(self.provider_identity)
        if type(self.requirements) not in {set, frozenset} or not _REQUIRED_ISOLATION.issubset(self.requirements):
            raise ContractError("isolation claim lacks required enforcement", code=FailureCode.SANDBOX_UNAVAILABLE)
        validate_sha256(self.attestation_digest)
        object.__setattr__(self, "requirements", frozenset(self.requirements))


@dataclass(frozen=True, slots=True, init=False)
class VerifiedIsolation:
    """Evidence issued by a separate evaluator-owned authority."""

    provider_id: str
    identity: str
    verifier_id: str
    verifier_identity: str
    requirements: frozenset[str]
    attestation_digest: str
    production_eligible: bool

    def __init__(
        self,
        provider_id: str,
        identity: str,
        verifier_id: str,
        verifier_identity: str,
        requirements: frozenset[str],
        attestation_digest: str,
        production_eligible: bool,
        *,
        _authority: "IsolationAuthority | None" = None,
    ) -> None:
        if _authority is None:
            raise ContractError("isolation evidence must be issued by an evaluator authority", code=FailureCode.SANDBOX_UNAVAILABLE)
        _non_empty_string(provider_id, label="provider_id")
        validate_sha256(identity)
        _non_empty_string(verifier_id, label="verifier_id")
        validate_sha256(verifier_identity)
        if type(requirements) not in {set, frozenset} or not _REQUIRED_ISOLATION.issubset(requirements):
            raise ContractError("isolation evidence lacks required enforcement", code=FailureCode.SANDBOX_UNAVAILABLE)
        validate_sha256(attestation_digest)
        if type(production_eligible) is not bool:
            raise ContractError("production_eligible must be a boolean", code=FailureCode.WRONG_TYPE)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "verifier_id", verifier_id)
        object.__setattr__(self, "verifier_identity", verifier_identity)
        object.__setattr__(self, "requirements", frozenset(requirements))
        object.__setattr__(self, "attestation_digest", attestation_digest)
        # G0 has no checked-in supervisor/authority.  Keep the field for wire
        # compatibility, but never let caller-created evidence opt into
        # production eligibility.
        object.__setattr__(self, "production_eligible", False)


@dataclass(frozen=True, slots=True)
class IsolatedProcess:
    process: subprocess.Popen[bytes]
    claim: IsolationClaim


class IsolationAuthority(ABC):
    """Out-of-band evaluator authority; providers cannot issue evidence."""

    def __init__(self, verifier_id: str, identity: str, *, production_eligible: bool = False) -> None:
        self._verifier_id = _non_empty_string(verifier_id, label="verifier_id")
        self._identity = validate_sha256(identity)
        if type(production_eligible) is not bool:
            raise ContractError("production_eligible must be a boolean", code=FailureCode.WRONG_TYPE)
        # This is deliberately not caller-configurable in G0.  A Python
        # object field is not authentication and cannot establish authority.
        self._production_eligible = False

    @property
    def verifier_id(self) -> str:
        return self._verifier_id

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def production_eligible(self) -> bool:
        return False

    @abstractmethod
    def verify(
        self,
        provider: "IsolationProvider",
        process: subprocess.Popen[bytes],
        claim: IsolationClaim,
    ) -> VerifiedIsolation:
        """Verify out-of-band enforcement and return evidence, or fail closed."""

    def _attest(self, provider: "IsolationProvider", claim: IsolationClaim) -> VerifiedIsolation:
        if claim.provider_id != provider.provider_id or claim.provider_identity != provider.identity:
            raise AutoMLXError("isolation claim provider identity mismatch", code=FailureCode.SANDBOX_UNAVAILABLE)
        return VerifiedIsolation(
            provider.provider_id,
            provider.identity,
            self.verifier_id,
            self.identity,
            claim.requirements,
            claim.attestation_digest,
            self.production_eligible,
            _authority=self,
        )


class IsolationProvider(ABC):
    """Evaluator-owned process launcher that returns only an untrusted claim.

    The default launch contract is deliberately unsupported.  A provider may
    opt in only when its synchronous ``enforce`` implementation gives the
    evaluator cleanup ownership before it can return.  ``execute_plan`` never
    runs this API in a worker thread that it does not itself bound and own,
    because a timed-out worker could publish a child after the evaluator had
    already failed closed.
    """

    def __init__(self, provider_id: str, identity: str, *, supports_evaluator_owned_launch: bool = False) -> None:
        self._provider_id = _non_empty_string(provider_id, label="provider_id")
        self._identity = validate_sha256(identity)
        if type(supports_evaluator_owned_launch) is not bool:
            raise ContractError("supports_evaluator_owned_launch must be a boolean", code=FailureCode.WRONG_TYPE)
        self._supports_evaluator_owned_launch = supports_evaluator_owned_launch

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def supports_evaluator_owned_launch(self) -> bool:
        return self._supports_evaluator_owned_launch

    def _claim(self, attestation_digest: str, requirements: frozenset[str] = _REQUIRED_ISOLATION) -> IsolationClaim:
        return IsolationClaim(self.provider_id, self.identity, requirements, attestation_digest)

    @abstractmethod
    def enforce(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str,
        env: Mapping[str, str],
        stdin: Any,
        stdout: Any,
        stderr: Any,
    ) -> IsolatedProcess:
        """Apply isolation and launch fixed argv, returning an untrusted claim."""


class UnavailableIsolationProvider(IsolationProvider):
    """Default provider for hosts without a real sandbox integration."""

    def __init__(self) -> None:
        super().__init__("unavailable", sha256_hex({"provider": "unavailable"}))

    def enforce(self, argv: tuple[str, ...], **kwargs: Any) -> IsolatedProcess:
        raise AutoMLXError(
            "no evaluator-owned isolation provider is available",
            code=FailureCode.SANDBOX_UNAVAILABLE,
        )


class UnavailableIsolationAuthority(IsolationAuthority):
    """Default authority for hosts without an out-of-band verifier."""

    def __init__(self) -> None:
        super().__init__("unavailable", sha256_hex({"verifier": "unavailable"}), production_eligible=False)

    def verify(self, provider: IsolationProvider, process: subprocess.Popen[bytes], claim: IsolationClaim) -> VerifiedIsolation:
        raise AutoMLXError(
            "no evaluator-owned isolation verifier is available",
            code=FailureCode.SANDBOX_UNAVAILABLE,
        )


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    timeout_seconds: float = 300.0
    max_stdout_bytes: int = 1_048_576
    max_stderr_bytes: int = 1_048_576
    max_output_bytes: int = 1_048_576
    kill_grace_seconds: float = 0.25
    require_network_denial: bool = True
    require_descendant_containment: bool = True
    temp_root: str | None = None
    extra_environment: Mapping[str, str] = field(default_factory=dict)
    # Retained for policy/wire compatibility.  G0 never invokes an external
    # launcher, so this value is not presented as a bound on arbitrary code.
    launch_timeout_seconds: float = 5.0
    authority_timeout_seconds: float = 5.0
    reader_join_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeout_seconds", _positive_number(self.timeout_seconds, label="timeout_seconds"))
        object.__setattr__(self, "kill_grace_seconds", _positive_number(self.kill_grace_seconds, label="kill_grace_seconds"))
        for name in ("launch_timeout_seconds", "authority_timeout_seconds", "reader_join_timeout_seconds"):
            object.__setattr__(self, name, _positive_number(getattr(self, name), label=name))
        for name in ("max_stdout_bytes", "max_stderr_bytes", "max_output_bytes"):
            _positive_int(getattr(self, name), label=name)
        if type(self.require_network_denial) is not bool or type(self.require_descendant_containment) is not bool:
            raise ContractError("isolation policy flags must be booleans", code=FailureCode.INVALID_POLICY)
        if self.temp_root is not None:
            _non_empty_string(self.temp_root, label="temp_root")
        object.__setattr__(self, "extra_environment", _freeze_environment(self.extra_environment))

    @property
    def required_isolation(self) -> frozenset[str]:
        required = set()
        if self.require_network_denial:
            required.add("network_denial")
        if self.require_descendant_containment:
            required.add("descendant_containment")
        return frozenset(required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "max_output_bytes": self.max_output_bytes,
            "kill_grace_seconds": self.kill_grace_seconds,
            "launch_timeout_seconds": self.launch_timeout_seconds,
            "authority_timeout_seconds": self.authority_timeout_seconds,
            "reader_join_timeout_seconds": self.reader_join_timeout_seconds,
            "require_network_denial": self.require_network_denial,
            "require_descendant_containment": self.require_descendant_containment,
            "temp_root": self.temp_root,
            "extra_environment": dict(self.extra_environment),
        }


def _open_runner_file(path: str | os.PathLike[str]) -> tuple[int, str]:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ContractError("runner artifact paths must be absolute", code=FailureCode.UNSAFE_PATH)
    try:
        if candidate.is_symlink():
            raise ContractError("runner artifact must not be a symlink", code=FailureCode.ARTIFACT_SYMLINK)
        if not hasattr(os, "O_NOFOLLOW"):
            raise ContractError("safe runner artifact access is unavailable", code=FailureCode.ARTIFACT_SYMLINK)
        stable = candidate.resolve(strict=True)
        flags = os.O_RDONLY | os.O_NOFOLLOW
        descriptor = os.open(str(stable), flags)
        stat_result = os.fstat(descriptor)
        if not stat.S_ISREG(stat_result.st_mode):
            os.close(descriptor)
            raise ContractError("runner artifact is not a regular file", code=FailureCode.ARTIFACT_NOT_REGULAR)
        return descriptor, str(stable)
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError("runner artifact could not be opened safely", code=FailureCode.ARTIFACT_MISSING) from exc


def _read_runner_bytes(path: str | os.PathLike[str], expected_size: int | None = None) -> tuple[int, str, str]:
    descriptor, stable_path = _open_runner_file(path)
    digest = hashlib.sha256()
    size = 0
    try:
        if expected_size is not None and expected_size > _MAX_REGISTERED_RUNNER_BYTES:
            raise ContractError(
                "runner artifact exceeds the configured size bound",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
        limit = _MAX_REGISTERED_RUNNER_BYTES if expected_size is None else expected_size
        remaining = limit
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ContractError(
                "runner artifact exceeds its declared/configured size bound",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
    except OSError as exc:
        raise ContractError("runner artifact could not be read safely", code=FailureCode.ARTIFACT_MISSING) from exc
    finally:
        os.close(descriptor)
    return size, digest.hexdigest(), stable_path


@dataclass(frozen=True, slots=True)
class RunnerArtifact:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not Path(self.path).is_absolute():
            raise ContractError("runner artifact path must be absolute", code=FailureCode.UNSAFE_PATH)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ContractError("runner artifact size must be non-negative", code=FailureCode.WRONG_TYPE)
        validate_sha256(self.sha256)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "RunnerArtifact":
        size, digest, stable_path = _read_runner_bytes(path)
        return cls(stable_path, size, digest)

    def verify(self) -> None:
        size, digest, _ = _read_runner_bytes(self.path, self.size_bytes)
        if size != self.size_bytes or digest != self.sha256:
            raise ContractError("runner artifact bytes changed after registration", code=FailureCode.IDENTITY_MISMATCH)


def _trusted_runner_digest(runner_id: str, argv: tuple[str, ...], artifacts: tuple[RunnerArtifact, ...]) -> str:
    return sha256_hex(
        {
            "runner_id": runner_id,
            "argv": list(argv),
            "artifacts": [
                {"path": item.path, "size_bytes": item.size_bytes, "sha256": item.sha256}
                for item in artifacts
            ],
        }
    )


@dataclass(frozen=True, slots=True, init=False)
class TrustedRunner:
    runner_id: str
    argv: tuple[str, ...]
    artifacts: tuple[RunnerArtifact, ...]
    digest: str

    def __init__(
        self,
        runner_id: str,
        argv: tuple[str, ...],
        artifacts: tuple[RunnerArtifact, ...],
    ) -> None:
        _non_empty_string(runner_id, label="runner_id")
        if type(argv) is not tuple or not argv or any(type(item) is not str or not item for item in argv):
            raise ContractError("trusted runner argv must be a non-empty string tuple", code=FailureCode.PROVIDER_ERROR)
        if type(artifacts) is not tuple or not artifacts:
            raise ContractError("trusted runner needs immutable artifact bindings", code=FailureCode.PROVIDER_ERROR)
        if any(not isinstance(item, RunnerArtifact) for item in artifacts):
            raise ContractError("trusted runner artifacts are malformed", code=FailureCode.WRONG_TYPE)
        bound_paths = {artifact.path for artifact in artifacts}
        if not Path(argv[0]).is_absolute() or argv[0] not in bound_paths:
            raise ContractError(
                "the runner executable must be an absolute path bound to immutable artifact bytes",
                code=FailureCode.IDENTITY_MISMATCH,
            )
        if any(Path(item).is_absolute() and item not in bound_paths for item in argv):
            raise ContractError(
                "every absolute runner file argument must be bound to immutable artifact bytes",
                code=FailureCode.IDENTITY_MISMATCH,
            )
        digest = _trusted_runner_digest(runner_id, argv, artifacts)
        object.__setattr__(self, "runner_id", runner_id)
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "digest", digest)

    @classmethod
    def from_command(
        cls,
        runner_id: str,
        argv: Sequence[str],
        *,
        artifact_paths: Sequence[str | os.PathLike[str]],
    ) -> "TrustedRunner":
        if type(argv) not in {list, tuple}:
            raise ContractError("trusted runner argv must be an array", code=FailureCode.WRONG_TYPE)
        if type(artifact_paths) not in {list, tuple} or not artifact_paths:
            raise ContractError("trusted runner artifact_paths must be a non-empty array", code=FailureCode.PROVIDER_ERROR)
        normalized_argv: list[str] = []
        for item in argv:
            candidate = Path(item)
            if candidate.is_absolute():
                try:
                    if not candidate.is_file():
                        raise ContractError(
                            "every absolute runner file argument must exist",
                            code=FailureCode.ARTIFACT_MISSING,
                        )
                    normalized_argv.append(str(candidate.resolve(strict=True)))
                except ContractError:
                    raise
                except OSError as exc:
                    raise ContractError(
                        "absolute runner file argument could not be resolved",
                        code=FailureCode.ARTIFACT_MISSING,
                    ) from exc
            else:
                normalized_argv.append(item)
        frozen_argv = tuple(normalized_argv)
        artifacts = tuple(RunnerArtifact.from_file(path) for path in artifact_paths)
        bound_paths = {artifact.path for artifact in artifacts}
        if not Path(frozen_argv[0]).is_absolute() or frozen_argv[0] not in bound_paths:
            raise ContractError(
                "the runner executable must be an absolute path bound to immutable artifact bytes",
                code=FailureCode.IDENTITY_MISMATCH,
            )
        for item in frozen_argv:
            candidate = Path(item)
            if candidate.is_absolute() and item not in bound_paths:
                raise ContractError(
                    "every absolute runner file argument must be bound to immutable artifact bytes",
                    code=FailureCode.IDENTITY_MISMATCH,
                )
        return cls(runner_id, frozen_argv, artifacts)

    def verify(self) -> None:
        for artifact in self.artifacts:
            artifact.verify()
        expected = _trusted_runner_digest(self.runner_id, self.argv, self.artifacts)
        if expected != self.digest:
            raise ContractError("trusted runner digest no longer matches its binding", code=FailureCode.IDENTITY_MISMATCH)


class TrustedRunnerRegistry:
    def __init__(self, runners: Sequence[TrustedRunner] = ()) -> None:
        self._runners: dict[str, TrustedRunner] = {}
        for runner in runners:
            self.register(runner)

    def register(self, runner: TrustedRunner) -> TrustedRunner:
        if not isinstance(runner, TrustedRunner):
            raise ContractError("registry accepts only TrustedRunner values", code=FailureCode.PROVIDER_ERROR)
        runner.verify()
        previous = self._runners.get(runner.runner_id)
        if previous is not None and previous != runner:
            raise ContractError("runner_id is already bound to a different digest", code=FailureCode.IDENTITY_MISMATCH)
        self._runners[runner.runner_id] = runner
        return runner

    def register_command(
        self,
        runner_id: str,
        argv: Sequence[str],
        *,
        artifact_paths: Sequence[str | os.PathLike[str]],
    ) -> TrustedRunner:
        return self.register(TrustedRunner.from_command(runner_id, argv, artifact_paths=artifact_paths))

    def resolve(self, runner_id: str) -> TrustedRunner:
        try:
            return self._runners[runner_id]
        except KeyError as exc:
            raise ContractError("runner is not registered", code=FailureCode.PROVIDER_ERROR) from exc

    @property
    def runners(self) -> Mapping[str, TrustedRunner]:
        return MappingProxyType(dict(self._runners))


def _execution_plan_digest(
    candidate_id: str,
    workload_hash: str,
    runner_id: str,
    runner_digest: str,
    runner_artifacts: tuple[RunnerArtifact, ...],
    artifacts: tuple[Artifact, ...],
    config: Mapping[str, Any],
) -> str:
    return sha256_hex(
        {
            "candidate_id": candidate_id,
            "workload_hash": workload_hash,
            "runner_id": runner_id,
            "runner_digest": runner_digest,
            "runner_artifacts": [
                {"path": item.path, "size_bytes": item.size_bytes, "sha256": item.sha256}
                for item in runner_artifacts
            ],
            "artifacts": [item.to_dict() for item in artifacts],
            "config": dict(config),
        }
    )


def _verify_declared_artifact(root: str, artifact: Artifact) -> None:
    descriptor = _open_verified_file(root, artifact.path)
    try:
        _verify_descriptor_bounded(descriptor, expected_size=artifact.size_bytes, expected_digest=artifact.sha256)
    finally:
        os.close(descriptor)


def _verify_execution_plan(plan: "ExecutionPlan", registry: TrustedRunnerRegistry) -> None:
    if not isinstance(registry, TrustedRunnerRegistry):
        raise ContractError("execution requires the evaluator TrustedRunnerRegistry", code=FailureCode.PROVIDER_ERROR)
    runner = registry.resolve(plan.runner_id)
    runner.verify()
    if plan.runner_digest != runner.digest or plan.argv != runner.argv or plan.runner_artifacts != runner.artifacts:
        raise ContractError(
            "execution plan runner binding does not match the registered runner",
            code=FailureCode.IDENTITY_MISMATCH,
        )
    expected = _execution_plan_digest(
        plan.candidate_id,
        plan.workload_hash,
        plan.runner_id,
        plan.runner_digest,
        plan.runner_artifacts,
        plan.artifacts,
        json_config(plan.config_bytes),
    )
    if expected != plan.plan_digest:
        raise ContractError("execution plan digest does not match its immutable fields", code=FailureCode.IDENTITY_MISMATCH)


def json_config(config_bytes: bytes) -> Mapping[str, Any]:
    """Decode only the evaluator's canonical config for plan re-binding."""
    try:
        value = strict_json_loads(config_bytes)
    except AutoMLXError as exc:
        raise ContractError("execution plan config bytes are not canonical JSON", code=FailureCode.CONFIG_MISMATCH) from exc
    if type(value) is not dict:
        raise ContractError("execution plan config must be a JSON object", code=FailureCode.CONFIG_MISMATCH)
    if canonical_bytes(value) != config_bytes:
        raise ContractError("execution plan config is not canonical", code=FailureCode.CONFIG_MISMATCH)
    return value


@dataclass(frozen=True, slots=True, init=False)
class ExecutionPlan:
    candidate_id: str
    workload_hash: str
    runner_id: str
    runner_digest: str
    argv: tuple[str, ...]
    runner_artifacts: tuple[RunnerArtifact, ...]
    artifacts: tuple[Artifact, ...]
    artifact_root: str
    config_bytes: bytes
    plan_digest: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        names = (
            "candidate_id", "workload_hash", "runner_id", "runner_digest", "argv",
            "runner_artifacts", "artifacts", "artifact_root", "config_bytes", "plan_digest",
        )
        if kwargs:
            if set(kwargs) - set(names):
                raise ContractError("execution plan has unknown fields", code=FailureCode.WRONG_TYPE)
            values = kwargs
        else:
            values = dict(zip(names, args, strict=True))
        for name in names:
            if name not in values:
                raise ContractError(f"execution plan missing {name}", code=FailureCode.WRONG_TYPE)
            value = values[name]
            if name in {"argv", "runner_artifacts", "artifacts"}:
                value = tuple(value)
            elif name == "config_bytes":
                value = bytes(value)
            object.__setattr__(self, name, value)

    def execute(
        self,
        policy: ExecutionPolicy,
        provider: IsolationProvider | None = None,
        *,
        registry: "TrustedRunnerRegistry | None" = None,
        authority: IsolationAuthority | None = None,
        observation_id: str | None = None,
        arm: str | None = None,
    ) -> "ExecutionRecord":
        return execute_plan(
            self,
            policy,
            registry=registry,
            provider=provider,
            authority=authority,
            observation_id=observation_id,
            arm=arm,
        )


def build_execution_plan(
    proposal: CandidateProposal,
    registry: TrustedRunnerRegistry,
    runner_id: str,
    artifact_root: str | os.PathLike[str],
) -> ExecutionPlan:
    if not isinstance(proposal, CandidateProposal):
        raise ContractError("execution plans require a CandidateProposal", code=FailureCode.WRONG_TYPE)
    if not isinstance(registry, TrustedRunnerRegistry):
        raise ContractError("execution plans require a TrustedRunnerRegistry", code=FailureCode.PROVIDER_ERROR)
    runner = registry.resolve(runner_id)
    runner.verify()
    root = str(Path(artifact_root).absolute())
    for artifact in proposal.workload.artifacts:
        _verify_declared_artifact(root, artifact)
    config_bytes = canonical_bytes(dict(proposal.config))
    digest = _execution_plan_digest(
        proposal.candidate_id,
        proposal.workload_hash,
        runner.runner_id,
        runner.digest,
        runner.artifacts,
        proposal.workload.artifacts,
        dict(proposal.config),
    )
    return ExecutionPlan(
        candidate_id=proposal.candidate_id,
        workload_hash=proposal.workload_hash,
        runner_id=runner.runner_id,
        runner_digest=runner.digest,
        argv=runner.argv,
        runner_artifacts=runner.artifacts,
        artifacts=proposal.workload.artifacts,
        artifact_root=root,
        config_bytes=config_bytes,
        plan_digest=digest,
    )


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    candidate_id: str
    workload_hash: str
    runner_id: str
    runner_digest: str
    status: ExecutionStatus
    parent_elapsed_ns: int
    runner_elapsed_ns: int | None = None
    observation_id: str | None = None
    arm: str | None = None
    returncode: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    output_truncated: bool = False
    isolation: VerifiedIsolation | None = None
    cleanup: CleanupObservation = field(default_factory=lambda: CleanupObservation(False, CleanupMode.NOT_NEEDED))
    failure: Failure | None = None

    def __post_init__(self) -> None:
        validate_sha256(self.candidate_id)
        validate_sha256(self.workload_hash)
        _non_empty_string(self.runner_id, label="runner_id")
        validate_sha256(self.runner_digest)
        if not isinstance(self.status, ExecutionStatus):
            raise ContractError("status must be an ExecutionStatus", code=FailureCode.WRONG_TYPE)
        if type(self.parent_elapsed_ns) is not int or self.parent_elapsed_ns < 0:
            raise ContractError("parent_elapsed_ns must be a non-negative integer", code=FailureCode.WRONG_TYPE)
        if self.runner_elapsed_ns is not None and (type(self.runner_elapsed_ns) is not int or self.runner_elapsed_ns <= 0):
            raise ContractError("runner_elapsed_ns must be a positive integer or null", code=FailureCode.WRONG_TYPE)
        if self.status is ExecutionStatus.SUCCESS and self.runner_elapsed_ns is None:
            raise ContractError("successful execution must record a runner_elapsed_ns", code=FailureCode.INVALID_VALUE)
        if self.observation_id is not None:
            _non_empty_string(self.observation_id, label="observation_id")
        if self.arm is not None and self.arm not in {"baseline", "candidate"}:
            raise ContractError("arm must be baseline, candidate, or null", code=FailureCode.WRONG_TYPE)
        if self.returncode is not None and type(self.returncode) is not int:
            raise ContractError("returncode must be an integer or null", code=FailureCode.WRONG_TYPE)
        if self.status is ExecutionStatus.SUCCESS and self.returncode != 0:
            raise ContractError("successful execution must have returncode 0", code=FailureCode.INVALID_VALUE)
        if self.status is ExecutionStatus.EXIT_FAILURE and (self.returncode is None or self.returncode == 0):
            raise ContractError("exit failure must have a non-zero returncode", code=FailureCode.INVALID_VALUE)
        if self.status is ExecutionStatus.CRASH and self.returncode is not None and self.returncode >= 0:
            raise ContractError("crash records must have a negative or null returncode", code=FailureCode.INVALID_VALUE)
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise ContractError("captured output must be bytes", code=FailureCode.WRONG_TYPE)
        for value in (self.stdout_truncated, self.stderr_truncated, self.output_truncated):
            if type(value) is not bool:
                raise ContractError("output flags must be booleans", code=FailureCode.WRONG_TYPE)
        if self.isolation is not None and type(self.isolation) is not VerifiedIsolation:
            raise ContractError("isolation must be provider-issued", code=FailureCode.WRONG_TYPE)
        if not isinstance(self.cleanup, CleanupObservation):
            raise ContractError("cleanup must be a CleanupObservation", code=FailureCode.WRONG_TYPE)
        if self.failure is not None and not isinstance(self.failure, Failure):
            raise ContractError("failure must be a Failure or null", code=FailureCode.WRONG_TYPE)

    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.SUCCESS

    @property
    def timing_ns(self) -> int:
        """The evidentiary timing quantity: the runner subprocess's own span.

        ``runner_elapsed_ns`` covers only launch-to-exit of the runner
        subprocess and excludes authority verification (three sandbox probe
        subprocesses, ~60ms) and artifact staging, both of which happen
        outside this span (see ``execute_plan``).  Falls back to
        ``parent_elapsed_ns`` (the full-sample diagnostic span) only when
        ``runner_elapsed_ns`` is unavailable, e.g. on a failure that never
        reached a real subprocess.
        """

        return self.runner_elapsed_ns if self.runner_elapsed_ns is not None else self.parent_elapsed_ns

    @property
    def output(self) -> bytes:
        return self.stdout

    @property
    def promotion_eligible(self) -> bool:
        """Whether this single record is safe to consume as promotable evidence.

        This is a structural, evidence-based check: real success, verified
        isolation attached, no truncation, no failure metadata, and cleanup
        that did not fail.  It intentionally never reads
        ``isolation.production_eligible`` -- that field is a caller-supplied
        claim (permanently withheld, see ``VerifiedIsolation``) and is not
        authentication.  A record built outside ``execute_plan`` with
        honestly-attested isolation and no failures is evidence-complete at
        this layer; it is not, by itself, a production activation decision.
        """

        return (
            self.succeeded
            and self.parent_elapsed_ns > 0
            and self.runner_elapsed_ns is not None
            and self.runner_elapsed_ns > 0
            and self.failure is None
            and not self.stdout_truncated
            and not self.stderr_truncated
            and not self.output_truncated
            and self.isolation is not None
            and self.cleanup.mode is not CleanupMode.FAILED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "workload_hash": self.workload_hash,
            "runner_id": self.runner_id,
            "runner_digest": self.runner_digest,
            "status": self.status.value,
            "parent_elapsed_ns": self.parent_elapsed_ns,
            "runner_elapsed_ns": self.runner_elapsed_ns,
            "observation_id": self.observation_id,
            "arm": self.arm,
            "returncode": self.returncode,
            "stdout_sha256": hashlib.sha256(self.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(self.stderr).hexdigest(),
            "stdout_bytes": len(self.stdout),
            "stderr_bytes": len(self.stderr),
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "output_truncated": self.output_truncated,
            "isolation": (
                {
                    "provider_id": self.isolation.provider_id,
                    "identity": self.isolation.identity,
                    "verifier_id": self.isolation.verifier_id,
                    "verifier_identity": self.isolation.verifier_identity,
                    "requirements": sorted(self.isolation.requirements),
                    "attestation_digest": self.isolation.attestation_digest,
                    "production_eligible": self.isolation.production_eligible,
                }
                if self.isolation
                else None
            ),
            "cleanup": {
                "attempted": self.cleanup.attempted,
                "mode": self.cleanup.mode.value,
                "verified": False,
            },
            "failure": self.failure.to_dict() if self.failure else None,
        }


class _BoundedCapture:
    def __init__(self, stdout_limit: int, stderr_limit: int, combined_limit: int) -> None:
        self._limits = (stdout_limit, stderr_limit)
        self._combined_limit = combined_limit
        self._buffers = [bytearray(), bytearray()]
        self._truncated = [False, False]
        self._combined_truncated = False
        self._total = 0
        self._errors: list[str] = []
        self._reader_states: list[str | None] = [None, None]
        self._lock = threading.Lock()
        self.output_event = threading.Event()
        self.failure_event = threading.Event()

    def append(self, stream_index: int, chunk: bytes) -> None:
        with self._lock:
            stream_remaining = max(0, self._limits[stream_index] - len(self._buffers[stream_index]))
            total_remaining = max(0, self._combined_limit - self._total)
            keep = min(len(chunk), stream_remaining, total_remaining)
            if keep:
                self._buffers[stream_index].extend(chunk[:keep])
                self._total += keep
            if keep != len(chunk):
                self._truncated[stream_index] = True
                self._combined_truncated = True
                self.output_event.set()

    def record_error(self, stream_index: int, error: BaseException) -> None:
        with self._lock:
            if 0 <= stream_index < len(self._reader_states) and self._reader_states[stream_index] == "error":
                return
            self._errors.append(f"stream-{stream_index}:{type(error).__name__}")
            if 0 <= stream_index < len(self._reader_states):
                self._reader_states[stream_index] = "error"
            self.failure_event.set()

    def record_completion(self, stream_index: int) -> None:
        with self._lock:
            if 0 <= stream_index < len(self._reader_states) and self._reader_states[stream_index] is None:
                self._reader_states[stream_index] = "complete"

    @property
    def reader_states(self) -> tuple[str | None, ...]:
        with self._lock:
            return tuple(self._reader_states)

    @property
    def errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._errors)

    def result(self) -> tuple[bytes, bytes, bool, bool, bool]:
        with self._lock:
            return (
                bytes(self._buffers[0]),
                bytes(self._buffers[1]),
                self._truncated[0],
                self._truncated[1],
                self._combined_truncated,
            )


def _read_pipe(pipe: Any, capture: _BoundedCapture, stream_index: int) -> None:
    try:
        while True:
            chunk = pipe.read(64 * 1024)
            if not chunk:
                return
            capture.append(stream_index, chunk)
    except BaseException as exc:
        capture.record_error(stream_index, exc)
        if isinstance(exc, KeyboardInterrupt):
            raise


def _read_pipe_worker(pipe: Any, capture: _BoundedCapture, stream_index: int) -> None:
    """Account for termination even if the reader target itself is replaced."""

    try:
        _read_pipe(pipe, capture, stream_index)
    except BaseException as exc:
        capture.record_error(stream_index, exc)
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        capture.record_completion(stream_index)


def _finish_capture(
    process: subprocess.Popen[bytes],
    readers: Sequence[threading.Thread],
    capture: _BoundedCapture,
    join_timeout_seconds: float,
) -> bool:
    """Drain and close both pipes, reporting every lifecycle ambiguity."""

    reader_failure = False
    for reader in readers:
        try:
            reader.join(timeout=join_timeout_seconds)
        except RuntimeError as exc:
            capture.record_error(-1, exc)
            reader_failure = True
    if any(reader.is_alive() for reader in readers):
        reader_failure = True
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception as exc:
                    capture.record_error(-1, exc)
        for reader in readers:
            try:
                reader.join(timeout=join_timeout_seconds)
            except RuntimeError as exc:
                capture.record_error(-1, exc)
    else:
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception as exc:
                    capture.record_error(-1, exc)
    if any(reader.is_alive() for reader in readers):
        capture.record_error(-1, RuntimeError("pipe reader did not terminate"))
        reader_failure = True
    for stream_index, state in enumerate(capture.reader_states):
        if state != "complete":
            if state is None:
                capture.record_error(stream_index, RuntimeError("pipe reader terminated without completion accounting"))
            reader_failure = True
    return reader_failure or bool(capture.errors)


def _terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float) -> CleanupObservation:
    """Best-effort cleanup only; this function never certifies containment."""

    try:
        if os.name == "posix":
            parent_group = os.getpgrp()
            try:
                process_group = os.getpgid(process.pid)
            except ProcessLookupError:
                process_group = None
            if process_group is not None and process_group != parent_group:
                os.killpg(process_group, signal.SIGTERM)
            elif process.poll() is None:
                # Never turn a missing/ambiguous session into a kill of the
                # evaluator's own process group.  A direct child signal is
                # less powerful, but remains target-specific.
                process.terminate()
            deadline = time.monotonic() + grace_seconds
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.005)
            if process.poll() is None:
                try:
                    if process_group is not None and process_group != parent_group:
                        os.killpg(process_group, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
        else:  # pragma: no cover - exercised only on Windows hosts.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        try:
            process.wait(timeout=max(grace_seconds, 0.05))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=max(grace_seconds, 0.05))
        if os.name == "posix":
            mode = CleanupMode.BEST_EFFORT_PROCESS_GROUP if process_group is not None and process_group != parent_group else CleanupMode.FAILED
        else:  # pragma: no cover - exercised only on Windows hosts.
            mode = CleanupMode.BEST_EFFORT_PROCESS_GROUP
        return CleanupObservation(True, mode)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        return CleanupObservation(True, CleanupMode.FAILED)


def _failure(code: FailureCode, message: str, **details: Any) -> Failure:
    return Failure(code, message, details)


def _record_failure(
    plan: ExecutionPlan,
    status: ExecutionStatus,
    failure: Failure,
    elapsed_ns: int,
    *,
    observation_id: str | None,
    arm: str | None,
    isolation: VerifiedIsolation | None = None,
    cleanup: CleanupObservation | None = None,
    returncode: int | None = None,
    stdout: bytes = b"",
    stderr: bytes = b"",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    output_truncated: bool = False,
    runner_elapsed_ns: int | None = None,
) -> ExecutionRecord:
    return ExecutionRecord(
        candidate_id=plan.candidate_id,
        workload_hash=plan.workload_hash,
        runner_id=plan.runner_id,
        runner_digest=plan.runner_digest,
        status=status,
        parent_elapsed_ns=max(1, elapsed_ns),
        runner_elapsed_ns=runner_elapsed_ns,
        observation_id=observation_id,
        arm=arm,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        output_truncated=output_truncated,
        isolation=isolation,
        cleanup=cleanup or CleanupObservation(False, CleanupMode.NOT_NEEDED),
        failure=failure,
    )


def _prepare_environment(workdir: Path, policy: ExecutionPolicy, config_path: Path, artifacts_path: Path) -> dict[str, str]:
    home = workdir / "home"
    cache = workdir / "cache"
    temp = workdir / "tmp"
    for path in (home, cache, temp):
        path.mkdir(mode=0o700)
    environment = {
        "PATH": os.defpath,
        "HOME": str(home),
        "TMPDIR": str(temp),
        "TEMP": str(temp),
        "TMP": str(temp),
        "XDG_CACHE_HOME": str(cache),
        "LANG": "C.UTF-8",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AUTO_MLX_CONFIG_PATH": str(config_path),
        "AUTO_MLX_ARTIFACT_ROOT": str(artifacts_path),
    }
    environment.update(policy.extra_environment)
    return environment


def _copy_descriptor_bounded(
    source_descriptor: int,
    destination: Path,
    *,
    expected_size: int,
    expected_digest: str,
    mode: int,
) -> None:
    """Copy exactly the frozen descriptor bytes, with the size bound enforced during I/O."""

    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    copied = 0
    try:
        destination_descriptor = os.open(
            str(destination),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            mode,
        )
        remaining = expected_size
        while remaining:
            chunk = os.read(source_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ContractError(
                    "frozen artifact ended before its expected size",
                    code=FailureCode.ARTIFACT_SIZE_MISMATCH,
                )
            digest.update(chunk)
            copied += len(chunk)
            remaining -= len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("artifact destination accepted no bytes")
                view = view[written:]
        if os.read(source_descriptor, 1):
            raise ContractError(
                "frozen artifact exceeded its expected size",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
        if copied != expected_size:
            raise ContractError(
                "frozen artifact size changed while being copied",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
        if digest.hexdigest() != expected_digest:
            raise ContractError(
                "frozen artifact digest changed while being copied",
                code=FailureCode.ARTIFACT_DIGEST_MISMATCH,
            )
        os.fchmod(destination_descriptor, mode)
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(
            "frozen artifact could not be copied safely",
            code=FailureCode.ARTIFACT_MISSING,
        ) from exc
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)


def _verify_descriptor_bounded(descriptor: int, *, expected_size: int, expected_digest: str) -> None:
    """Verify a staged descriptor without reading beyond the frozen size budget."""

    digest = hashlib.sha256()
    remaining = expected_size
    copied = 0
    try:
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ContractError(
                    "staged artifact ended before its expected size",
                    code=FailureCode.ARTIFACT_SIZE_MISMATCH,
                )
            digest.update(chunk)
            copied += len(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ContractError(
                "staged artifact exceeded its expected size",
                code=FailureCode.ARTIFACT_SIZE_MISMATCH,
            )
    except OSError as exc:
        raise ContractError("staged artifact could not be verified", code=FailureCode.ARTIFACT_MISSING) from exc
    if copied != expected_size:
        raise ContractError("staged artifact size does not match its binding", code=FailureCode.ARTIFACT_SIZE_MISMATCH)
    if digest.hexdigest() != expected_digest:
        raise ContractError("staged artifact digest does not match its binding", code=FailureCode.ARTIFACT_DIGEST_MISMATCH)


def _verify_staged_file(path: Path, *, expected_size: int, expected_digest: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ContractError("staged artifact is not a regular file", code=FailureCode.ARTIFACT_NOT_REGULAR)
        _verify_descriptor_bounded(descriptor, expected_size=expected_size, expected_digest=expected_digest)
    finally:
        os.close(descriptor)


def _copy_frozen_artifacts(plan: ExecutionPlan, artifacts_path: Path) -> None:
    for artifact in plan.artifacts:
        destination = artifacts_path / artifact.path
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        source_descriptor = _open_verified_file(plan.artifact_root, artifact.path)
        try:
            _copy_descriptor_bounded(
                source_descriptor,
                destination,
                expected_size=artifact.size_bytes,
                expected_digest=artifact.sha256,
                mode=0o400,
            )
        finally:
            os.close(source_descriptor)
        _verify_staged_file(destination, expected_size=artifact.size_bytes, expected_digest=artifact.sha256)


def _stage_runner_artifacts(plan: ExecutionPlan, runner_path: Path) -> tuple[str, ...]:
    """Stage every registered runner file and rewrite argv to those pinned copies."""

    runner_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    replacements: dict[str, str] = {}
    for index, artifact in enumerate(plan.runner_artifacts):
        source_descriptor, stable_path = _open_runner_file(artifact.path)
        destination = runner_path / f"{index:04d}-{Path(stable_path).name}"
        try:
            source_mode = stat.S_IMODE(os.fstat(source_descriptor).st_mode)
            mode = source_mode | stat.S_IRUSR
            _copy_descriptor_bounded(
                source_descriptor,
                destination,
                expected_size=artifact.size_bytes,
                expected_digest=artifact.sha256,
                mode=mode,
            )
        finally:
            os.close(source_descriptor)
        _verify_staged_file(destination, expected_size=artifact.size_bytes, expected_digest=artifact.sha256)
        replacements[artifact.path] = str(destination)
    expected = _trusted_runner_digest(plan.runner_id, plan.argv, plan.runner_artifacts)
    if expected != plan.runner_digest:
        raise ContractError("execution runner bytes changed after plan construction", code=FailureCode.IDENTITY_MISMATCH)
    return tuple(replacements.get(argument, argument) for argument in plan.argv)


def execute_plan(
    plan: ExecutionPlan,
    policy: ExecutionPolicy,
    *,
    registry: TrustedRunnerRegistry | None = None,
    provider: IsolationProvider | None = None,
    authority: IsolationAuthority | None = None,
    observation_id: str | None = None,
    arm: str | None = None,
) -> ExecutionRecord:
    """Execute only through an evaluator-owned isolation provider.

    This is a host-primitive gate followed by a caller opt-in gate, both
    fail-closed:

    1. Without the local sandbox primitives (macOS, ``sandbox-exec`` on
       PATH, descriptor-relative artifact access) this always returns
       ``SANDBOX_UNAVAILABLE`` -- identical to a host that has no execution
       engine at all.  This keeps non-macOS hosts (including Linux CI)
       behavior-identical to the G0 stub regardless of which provider a
       caller passes.
    2. With the primitives present, a caller must still explicitly
       construct and pass a concrete ``provider``/``authority`` pair; the
       module-level defaults (``UnavailableIsolationProvider`` /
       ``UnavailableIsolationAuthority`` / ``None``) remain fail-closed.
       Real execution is always an explicit caller decision, never a
       default.
    """

    if not isinstance(plan, ExecutionPlan):
        raise ContractError("execute_plan requires an ExecutionPlan", code=FailureCode.WRONG_TYPE)
    if not isinstance(policy, ExecutionPolicy):
        raise ContractError("execute_plan requires an ExecutionPolicy", code=FailureCode.WRONG_TYPE)
    if observation_id is not None:
        _non_empty_string(observation_id, label="observation_id")
    if arm is not None and arm not in {"baseline", "candidate"}:
        raise ContractError("arm must be baseline, candidate, or null", code=FailureCode.WRONG_TYPE)
    started_ns = time.monotonic_ns()

    if not local_sandbox_primitives_available():
        return _record_failure(
            plan,
            ExecutionStatus.SANDBOX_UNAVAILABLE,
            _failure(
                FailureCode.SANDBOX_UNAVAILABLE,
                "this host lacks the local sandbox execution primitives (macOS + sandbox-exec + descriptor-relative artifact access)",
            ),
            time.monotonic_ns() - started_ns,
            observation_id=observation_id,
            arm=arm,
        )
    if not isinstance(provider, IsolationProvider):
        return _record_failure(
            plan,
            ExecutionStatus.SANDBOX_UNAVAILABLE,
            _failure(
                FailureCode.SANDBOX_UNAVAILABLE,
                "an evaluator-owned isolation provider is required; process groups are not sandbox proof",
            ),
            time.monotonic_ns() - started_ns,
            observation_id=observation_id,
            arm=arm,
        )
    if not isinstance(authority, IsolationAuthority):
        return _record_failure(
            plan,
            ExecutionStatus.SANDBOX_UNAVAILABLE,
            _failure(FailureCode.SANDBOX_UNAVAILABLE, "an evaluator-owned isolation authority is required; provider claims are not evidence"),
            time.monotonic_ns() - started_ns,
            observation_id=observation_id,
            arm=arm,
        )
    if not _REQUIRED_ISOLATION.issubset(policy.required_isolation):
        return _record_failure(
            plan,
            ExecutionStatus.SANDBOX_UNAVAILABLE,
            _failure(FailureCode.SANDBOX_UNAVAILABLE, "evaluation policy must require network and descendant isolation"),
            time.monotonic_ns() - started_ns,
            observation_id=observation_id,
            arm=arm,
        )
    if not provider.supports_evaluator_owned_launch:
        return _record_failure(
            plan,
            ExecutionStatus.SANDBOX_UNAVAILABLE,
            _failure(
                FailureCode.SANDBOX_UNAVAILABLE,
                "isolation provider launch contract does not provide evaluator-owned cleanup before invocation",
            ),
            time.monotonic_ns() - started_ns,
            observation_id=observation_id,
            arm=arm,
        )

    process: subprocess.Popen[bytes] | None = None
    capture: _BoundedCapture | None = None
    readers: list[threading.Thread] = []
    isolation: VerifiedIsolation | None = None
    cleanup = CleanupObservation(False, CleanupMode.NOT_NEEDED)
    timed_out = False
    output_failed = False
    returncode: int | None = None
    try:
        try:
            _verify_execution_plan(plan, registry)  # type: ignore[arg-type]
        except AutoMLXError as exc:
            return _record_failure(
                plan,
                ExecutionStatus.ARTIFACT_FAILURE,
                Failure(exc.code, str(exc)),
                time.monotonic_ns() - started_ns,
                observation_id=observation_id,
                arm=arm,
            )
        with tempfile.TemporaryDirectory(prefix=".auto-mlx-", dir=policy.temp_root) as temporary:
            workdir = Path(temporary)
            artifacts_path = workdir / _ARTIFACT_DIRECTORY
            artifacts_path.mkdir(mode=0o700)
            try:
                runner_argv = _stage_runner_artifacts(plan, workdir / ".auto_mlx_runner")
            except AutoMLXError as exc:
                return _record_failure(
                    plan,
                    ExecutionStatus.ARTIFACT_FAILURE,
                    Failure(exc.code, str(exc)),
                    time.monotonic_ns() - started_ns,
                    observation_id=observation_id,
                    arm=arm,
                )
            config_path = workdir / _CONFIG_FILE
            config_path.write_bytes(plan.config_bytes)
            os.chmod(config_path, 0o400)
            try:
                _copy_frozen_artifacts(plan, artifacts_path)
            except AutoMLXError as exc:
                return _record_failure(
                    plan,
                    ExecutionStatus.ARTIFACT_FAILURE,
                    Failure(exc.code, str(exc)),
                    time.monotonic_ns() - started_ns,
                    observation_id=observation_id,
                    arm=arm,
                )
            environment = _prepare_environment(workdir, policy, config_path, artifacts_path)
            try:
                launched = provider.enforce(
                    runner_argv,
                    cwd=str(workdir),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as exc:
                return _record_failure(
                    plan,
                    ExecutionStatus.SANDBOX_UNAVAILABLE,
                    _failure(
                        FailureCode.TIMEOUT if isinstance(exc, _HandshakeTimeout) else FailureCode.SANDBOX_UNAVAILABLE,
                        "isolation provider handshake did not complete safely",
                        error=type(exc).__name__,
                    ),
                    time.monotonic_ns() - started_ns,
                    observation_id=observation_id,
                    arm=arm,
                )
            if not isinstance(launched, IsolatedProcess) or not isinstance(launched.process, subprocess.Popen):
                return _record_failure(
                    plan,
                    ExecutionStatus.SANDBOX_UNAVAILABLE,
                    _failure(FailureCode.SANDBOX_UNAVAILABLE, "isolation provider returned no verified process"),
                    time.monotonic_ns() - started_ns,
                    observation_id=observation_id,
                    arm=arm,
                )
            process = launched.process
            if process.stdout is None or process.stderr is None:
                cleanup = _terminate_process_group(process, policy.kill_grace_seconds)
                return _record_failure(
                    plan,
                    ExecutionStatus.START_FAILURE,
                    _failure(FailureCode.RUNTIME_FAILURE, "isolation provider did not attach bounded output pipes"),
                    time.monotonic_ns() - started_ns,
                    observation_id=observation_id,
                    arm=arm,
                    isolation=isolation,
                    cleanup=cleanup,
                )
            capture = _BoundedCapture(policy.max_stdout_bytes, policy.max_stderr_bytes, policy.max_output_bytes)
            readers = [
                threading.Thread(target=_read_pipe_worker, args=(process.stdout, capture, 0), daemon=True),
                threading.Thread(target=_read_pipe_worker, args=(process.stderr, capture, 1), daemon=True),
            ]
            for reader in readers:
                reader.start()
            # The evidentiary timed span starts here (the runner subprocess
            # is launched and its pipes are being drained) and ends the
            # instant its exit is observed below.  Authority verification is
            # deliberately run AFTER this span closes (see the comment near
            # ``authority.verify`` below) so its ~3-probe, ~60ms cost never
            # pollutes the measured quantity.
            runner_started_ns = time.monotonic_ns()

            deadline = time.monotonic() + policy.timeout_seconds
            while process.poll() is None:
                if capture.output_event.is_set() or capture.failure_event.is_set():
                    output_failed = True
                    cleanup = _terminate_process_group(process, policy.kill_grace_seconds)
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    cleanup = _terminate_process_group(process, policy.kill_grace_seconds)
                    break
                time.sleep(0.005)
            if (capture.output_event.is_set() or capture.failure_event.is_set()) and not output_failed:
                output_failed = True
                cleanup = _terminate_process_group(process, policy.kill_grace_seconds)
            returncode = process.poll()
            if returncode is None:
                try:
                    returncode = process.wait(timeout=policy.kill_grace_seconds)
                except subprocess.TimeoutExpired:
                    cleanup = _terminate_process_group(process, policy.kill_grace_seconds)
                    returncode = process.poll()
            # Runner span closes here: the process is confirmed exited
            # (naturally, or terminated above), before authority
            # verification's probe subprocesses run.  This is the sole
            # evidentiary duration; downstream measurement/gain math must
            # read ``runner_elapsed_ns``, never ``parent_elapsed_ns``.
            runner_elapsed_ns = max(1, time.monotonic_ns() - runner_started_ns)
            reader_failure = _finish_capture(process, readers, capture, policy.reader_join_timeout_seconds)
            stdout, stderr, stdout_truncated, stderr_truncated, output_truncated = capture.result()

            # Verification now runs after the runner's own subprocess
            # lifetime, never overlapping the timed span above.
            # ``LocalSandboxAuthority.verify`` (auto_mlx.sandbox) recovers
            # the profile text from the immutable, already-set
            # ``Popen.args`` attribute (valid before or after the process
            # exits) and launches its own fresh probe subprocesses under
            # that profile; it never reads live-process state (no stdout/
            # stdin interaction, no reliance on the process still running).
            # Moving the call here therefore changes nothing about the
            # evidence it produces -- only when its cost is paid relative to
            # the evidentiary clock.  A verification failure still
            # unconditionally overrides any process-level outcome below,
            # exactly as when verification ran first.
            try:
                isolation = _call_bounded(
                    lambda: authority.verify(provider, process, launched.claim),
                    policy.authority_timeout_seconds,
                )
            except Exception as exc:
                return _record_failure(
                    plan,
                    ExecutionStatus.SANDBOX_UNAVAILABLE,
                    _failure(
                        FailureCode.TIMEOUT if isinstance(exc, _HandshakeTimeout) else FailureCode.SANDBOX_UNAVAILABLE,
                        "isolation authority could not verify provider enforcement",
                        error=type(exc).__name__,
                        capture_errors=list(capture.errors),
                        reader_failure=reader_failure,
                    ),
                    time.monotonic_ns() - started_ns,
                    observation_id=observation_id,
                    arm=arm,
                    cleanup=cleanup,
                    stdout=stdout,
                    stderr=stderr,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                    output_truncated=output_truncated,
                    runner_elapsed_ns=runner_elapsed_ns,
                )
            if (
                not isinstance(isolation, VerifiedIsolation)
                or isolation.provider_id != provider.provider_id
                or isolation.identity != provider.identity
                or isolation.verifier_id != authority.verifier_id
                or isolation.verifier_identity != authority.identity
                or not policy.required_isolation.issubset(isolation.requirements)
            ):
                return _record_failure(
                    plan,
                    ExecutionStatus.SANDBOX_UNAVAILABLE,
                    _failure(
                        FailureCode.SANDBOX_UNAVAILABLE,
                        "isolation authority returned mismatched evidence identity",
                        capture_errors=list(capture.errors),
                        reader_failure=reader_failure,
                    ),
                    time.monotonic_ns() - started_ns,
                    observation_id=observation_id,
                    arm=arm,
                    cleanup=cleanup,
                    stdout=stdout,
                    stderr=stderr,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                    output_truncated=output_truncated,
                    runner_elapsed_ns=runner_elapsed_ns,
                )
            elapsed_ns = max(1, time.monotonic_ns() - started_ns)
            common = {
                "observation_id": observation_id,
                "arm": arm,
                "isolation": isolation,
                "cleanup": cleanup,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "output_truncated": output_truncated,
                "runner_elapsed_ns": runner_elapsed_ns,
            }
            if output_failed or output_truncated or reader_failure:
                return _record_failure(
                    plan,
                    ExecutionStatus.OUTPUT_FAILURE,
                    _failure(
                        FailureCode.OUTPUT_LIMIT if output_truncated else FailureCode.RUNTIME_FAILURE,
                        "runner output capture did not complete safely",
                        capture_errors=list(capture.errors),
                        reader_failure=reader_failure,
                    ),
                    elapsed_ns,
                    **common,
                )
            if timed_out:
                return _record_failure(
                    plan,
                    ExecutionStatus.TIMEOUT,
                    _failure(FailureCode.TIMEOUT, "runner exceeded the evaluator timeout", timeout_seconds=policy.timeout_seconds),
                    elapsed_ns,
                    **common,
                )
            if returncode is None or returncode < 0:
                return _record_failure(
                    plan,
                    ExecutionStatus.CRASH,
                    _failure(FailureCode.RUNTIME_FAILURE, "runner terminated by a signal", returncode=returncode),
                    elapsed_ns,
                    **common,
                )
            if returncode != 0:
                return _record_failure(
                    plan,
                    ExecutionStatus.EXIT_FAILURE,
                    _failure(FailureCode.RUNTIME_FAILURE, "runner exited unsuccessfully", returncode=returncode),
                    elapsed_ns,
                    **common,
                )
            if cleanup.mode is CleanupMode.FAILED:
                return _record_failure(
                    plan,
                    ExecutionStatus.START_FAILURE,
                    _failure(FailureCode.RUNTIME_FAILURE, "runner cleanup was ambiguous"),
                    elapsed_ns,
                    **common,
                )
            return ExecutionRecord(
                candidate_id=plan.candidate_id,
                workload_hash=plan.workload_hash,
                runner_id=plan.runner_id,
                runner_digest=plan.runner_digest,
                status=ExecutionStatus.SUCCESS,
                parent_elapsed_ns=elapsed_ns,
                runner_elapsed_ns=runner_elapsed_ns,
                observation_id=observation_id,
                arm=arm,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                isolation=isolation,
                cleanup=cleanup,
            )
    except Exception as exc:
        if process is not None and process.poll() is None:
            cleanup = _terminate_process_group(process, policy.kill_grace_seconds)
        if process is not None and capture is not None:
            _finish_capture(process, readers, capture, policy.reader_join_timeout_seconds)
            stdout, stderr, stdout_truncated, stderr_truncated, output_truncated = capture.result()
        else:
            stdout = stderr = b""
            stdout_truncated = stderr_truncated = output_truncated = False
        return _record_failure(
            plan,
            ExecutionStatus.START_FAILURE,
            _failure(
                FailureCode.RUNTIME_FAILURE,
                "executor failed closed after an unexpected error",
                error=type(exc).__name__,
                capture_errors=list(capture.errors) if capture is not None else [],
            ),
            time.monotonic_ns() - started_ns,
            observation_id=observation_id,
            arm=arm,
            isolation=isolation,
            cleanup=cleanup,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            output_truncated=output_truncated,
        )


__all__: Final = [
    "CleanupMode",
    "CleanupObservation",
    "ExecutionPlan",
    "ExecutionPolicy",
    "ExecutionRecord",
    "ExecutionStatus",
    "IsolationAuthority",
    "IsolationClaim",
    "IsolationProvider",
    "IsolatedProcess",
    "RunnerArtifact",
    "TrustedRunner",
    "TrustedRunnerRegistry",
    "UnavailableIsolationAuthority",
    "UnavailableIsolationProvider",
    "VerifiedIsolation",
    "build_execution_plan",
    "execute_plan",
    "local_sandbox_primitives_available",
]
