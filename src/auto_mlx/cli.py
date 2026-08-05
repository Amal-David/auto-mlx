"""MLX-free command line interface for Auto MLX.

The CLI validates declarative documents and inspects identities without ever
importing MLX itself.  ``evaluate``, ``promote``, ``dispatch``, ``rollback``,
and ``keys ensure`` wire the real local evidence-gated loop -- sandboxed
execution, receipt construction, local supervisor attestation, promotion
decisions, and dispatch -- through the local sandbox tier
(:mod:`auto_mlx.sandbox`).  On a host without the local sandbox execution
primitives (non-macOS, no ``sandbox-exec``, ...), ``evaluate`` and
``dispatch --execute`` fail closed with a stable ``unavailable`` diagnostic
and exit code 4, identical in kind to every other fail-closed boundary in
this codebase; ``promote``, plain ``dispatch``, ``rollback``, and
``keys ensure`` never need the sandbox and always run.  Real production
(G3) activation remains a separate, later gate -- see
``docs/evidence-and-promotion.md``.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
from importlib.metadata import PackageNotFoundError, version as distribution_version
import os
import secrets
import stat
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Final

from .canonical import canonical_json, sha256_hex, strict_json_loads
from .contracts import (
    Artifact,
    CandidateProposal,
    EvaluationPolicy,
    FrozenWorkload,
    Knob,
    RuntimeIdentity,
)
from .dispatch import CANDIDATE_MODE, DEFAULT_MAX_AGE_NS, NATIVE_MODE
from .dispatch import dispatch as run_dispatch
from .errors import AutoMLXError, CanonicalJSONError, ContractError, FailureCode, KeyMaterialError, SupervisorRefusalError
from .evaluator import Evaluator
from .executor import (
    ExecutionPolicy,
    ExecutionStatus,
    TrustedRunnerRegistry,
    build_execution_plan,
    local_sandbox_primitives_available,
)
from . import keys as keys_module
from .oracle import ExactOutputOracle
from .promotion import activate as activate_decision
from .promotion import rollback as rollback_decision
from .providers import DeclarativeProvider
from .receipts import Receipt, validate_receipt
from .runners import BASELINE_RUNNER_ID, CANDIDATE_RUNNER_ID, register_reference_matmul_runners
from .sandbox import LocalSandboxAuthority, LocalSandboxProvider
from . import store_config
from .supervisor import attest_receipt
from . import tune as tune_module


EXIT_OK: Final = 0
EXIT_USAGE: Final = 2
EXIT_CONTRACT: Final = 3
EXIT_UNAVAILABLE: Final = 4
EXIT_IO: Final = 5
EXIT_INTERNAL: Final = 70
MAX_JSON_INPUT_BYTES: Final = 4 * 1024 * 1024

_OPEN_SUPPORTS_DIR_FD: Final = os.open in getattr(os, "supports_dir_fd", ())
_STAT_SUPPORTS_DIR_FD: Final = os.stat in getattr(os, "supports_dir_fd", ())
_STAT_SUPPORTS_NOFOLLOW: Final = os.stat in getattr(os, "supports_follow_symlinks", ())
_LINK_SUPPORTS_DIR_FD: Final = os.link in getattr(os, "supports_dir_fd", ())
_LINK_SUPPORTS_NOFOLLOW: Final = os.link in getattr(os, "supports_follow_symlinks", ())
_UNLINK_SUPPORTS_DIR_FD: Final = os.unlink in getattr(os, "supports_dir_fd", ())
_NON_REGULAR_OPEN_ERRNOS: Final = frozenset(
    value
    for value in (
        errno.EISDIR,
        errno.ENOTDIR,
        errno.ENXIO,
        errno.EOPNOTSUPP,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "ENODEV", None),
        getattr(errno, "ENOTTY", None),
    )
    if value is not None
)

_CONTRACT_KINDS: Final = (
    "artifact",
    "candidate",
    "knob",
    "policy",
    "provider",
    "receipt",
    "runtime",
    "workload",
    "document",
)


class CLIUsageError(Exception):
    """An argument error that can be rendered as the CLI's JSON diagnostic."""


class CLIIOError(Exception):
    """An input/output failure at the CLI boundary."""


class CLIUnavailableError(Exception):
    """A recognized, real command that cannot run on this host right now.

    Raised only for the local-sandbox-primitives gate (see
    :func:`auto_mlx.executor.local_sandbox_primitives_available`) -- never for
    a command that is merely unimplemented.  Renders as the same stable
    ``unavailable`` JSON diagnostic and exit code 4 that this CLI has always
    used for a recognized-but-unrunnable command.
    """

    def __init__(self, message: str, *, stage: str, surface: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.surface = surface


# Workload name -> a CLI-owned function that registers that workload's
# trusted runner(s) into a fresh registry and returns
# ``(baseline_runner_id, candidate_runner_id)``.  This binding is CLI-owned
# and closed: a candidate proposal's config never selects a runner or
# command, only knob values within a workload the CLI already knows how to
# run.  An unrecognized workload name is a typed, fail-closed diagnostic
# (see ``_resolve_workload_runners``), never a generic fallback runner.
def _register_toy_matmul_runners(registry: TrustedRunnerRegistry) -> tuple[str, str]:
    register_reference_matmul_runners(registry)
    return BASELINE_RUNNER_ID, CANDIDATE_RUNNER_ID


_WORKLOAD_RUNNERS: Final[dict[str, Callable[[TrustedRunnerRegistry], tuple[str, str]]]] = {
    "toy-matmul": _register_toy_matmul_runners,
}


def _resolve_workload_runners(workload: FrozenWorkload, registry: TrustedRunnerRegistry) -> tuple[str, str]:
    registrar = _WORKLOAD_RUNNERS.get(workload.name)
    if registrar is None:
        raise ContractError(
            f"no trusted runner is registered for workload {workload.name!r}",
            code=FailureCode.PROVIDER_ERROR,
        )
    return registrar(registry)


def _execution_policy_from_evaluation_policy(policy: EvaluationPolicy) -> ExecutionPolicy:
    """Mirror :func:`auto_mlx.evaluator._execution_policy_from_contract`.

    Duplicated (not imported) because that helper is a private, evaluator-
    internal symbol; this CLI-owned copy is used for the one-off baseline
    probe and ``dispatch --execute`` runs that happen outside an
    :class:`auto_mlx.evaluator.Evaluator` instance.  ``Evaluator`` itself
    still derives and checks its own execution policy independently.
    """

    return ExecutionPolicy(
        timeout_seconds=float(policy.timeout_seconds),
        max_stdout_bytes=policy.max_output_bytes,
        max_stderr_bytes=policy.max_output_bytes,
        max_output_bytes=policy.max_output_bytes,
        require_network_denial=True,
        require_descendant_containment=True,
        extra_environment={"AUTO_MLX_K_REPETITIONS": str(policy.k_repetitions)},
    )


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(message)


def _kind_arguments(parser: argparse.ArgumentParser, *, command: str) -> None:
    parser.add_argument("kind", nargs="?", choices=_CONTRACT_KINDS, help="document kind")
    parser.add_argument(
        "path",
        nargs="?",
        help="document path, or '-' for stdin (the --input spelling is also supported)",
    )
    parser.add_argument("--kind", "--type", dest="kind_option", choices=_CONTRACT_KINDS, help=argparse.SUPPRESS)
    parser.add_argument("-i", "--input", dest="input_option", help="document path, or '-' for stdin")
    parser.add_argument("--workload", help="workload document required for candidate validation")
    parser.add_argument("--artifact-root", help="local root against which workload artifact bytes are checked")
    if command == "validate":
        parser.add_argument("--output", help="create this canonical document file; an existing file is never overwritten")


def _package_version() -> str:
    # When running from a checkout (including an editable install), prefer the
    # adjacent project metadata so a stale installed distribution cannot mask
    # the source tree being exercised.
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with project_file.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except FileNotFoundError:
        pass
    except (KeyError, OSError, TypeError, ValueError):
        return "unknown"
    try:
        return distribution_version("auto-mlx")
    except PackageNotFoundError:
        return "unknown"


def _context_document_arguments(
    parser: argparse.ArgumentParser, *, policy_overrides: bool, artifact_root_required: bool
) -> None:
    """Flags shared by ``evaluate`` and ``dispatch`` for their evaluation context."""

    parser.add_argument("--workload", required=True, help="workload document path")
    parser.add_argument("--candidate", required=True, help="candidate proposal document path")
    parser.add_argument("--policy", help="evaluation policy document path (defaults to policy defaults)")
    parser.add_argument(
        "--runtime",
        help="runtime identity document path; if given, must match this host's current runtime identity "
        "(defaults to this host's current runtime identity)",
    )
    if artifact_root_required:
        parser.add_argument(
            "--artifact-root",
            required=True,
            help="local root the evaluator reads and verifies workload artifacts against",
        )
    else:
        parser.add_argument(
            "--artifact-root",
            help="local root used to verify workload artifacts, and to run --execute (defaults to the current directory)",
        )
    parser.add_argument("--store", help="receipt/decision store root (defaults to AUTO_MLX_STORE or ./auto-mlx-store)")
    parser.add_argument(
        "--key-dir", help="attestation key directory (defaults to AUTO_MLX_KEY_DIR or ~/.auto-mlx/keys)"
    )
    if policy_overrides:
        parser.add_argument(
            "--samples", type=int, dest="measurement_runs", help="override policy.measurement_runs (sequential sampling start)"
        )
        parser.add_argument("--warmup-runs", type=int, help="override policy.warmup_runs")
        parser.add_argument("--timeout-seconds", type=int, help="override policy.timeout_seconds")
        parser.add_argument("--max-output-bytes", type=int, help="override policy.max_output_bytes")
        parser.add_argument("--k-repetitions", type=int, help="override policy.k_repetitions (in-runner timed iterations)")
        parser.add_argument(
            "--max-measurement-runs", type=int, help="override policy.max_measurement_runs (sequential sampling cap)"
        )
        parser.add_argument(
            "--min-effect-bps", type=int, help="override policy.min_effect_bps (promotion threshold, basis points)"
        )
        parser.add_argument("--bootstrap-resamples", type=int, help="override policy.bootstrap_resamples")


def _tune_context_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for ``tune``: like ``evaluate``'s context, but a provider grid, not one candidate."""

    parser.add_argument("--workload", required=True, help="workload document path")
    parser.add_argument(
        "--provider", required=True, help="declarative provider document path (the knob grid to race)"
    )
    parser.add_argument("--policy", help="evaluation policy document path (defaults to policy defaults)")
    parser.add_argument(
        "--runtime",
        help="runtime identity document path; if given, must match this host's current runtime identity "
        "(defaults to this host's current runtime identity)",
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="local root the evaluator reads and verifies workload artifacts against",
    )
    parser.add_argument(
        "--store", help="receipt/decision/tuning-summary store root (defaults to AUTO_MLX_STORE or ./auto-mlx-store)"
    )
    parser.add_argument(
        "--key-dir", help="attestation key directory (defaults to AUTO_MLX_KEY_DIR or ~/.auto-mlx/keys)"
    )
    parser.add_argument(
        "--samples", type=int, dest="measurement_runs",
        help="override policy.measurement_runs (starting/minimum blocks per racing rung)",
    )
    parser.add_argument("--warmup-runs", type=int, help="override policy.warmup_runs")
    parser.add_argument("--timeout-seconds", type=int, help="override policy.timeout_seconds")
    parser.add_argument("--max-output-bytes", type=int, help="override policy.max_output_bytes")
    parser.add_argument("--k-repetitions", type=int, help="override policy.k_repetitions (in-runner timed iterations)")
    parser.add_argument(
        "--max-measurement-runs", type=int,
        help="override policy.max_measurement_runs (per-candidate racing ladder cap)",
    )
    parser.add_argument(
        "--min-effect-bps", type=int,
        help="override policy.min_effect_bps (promotion and racing-futility threshold, basis points)",
    )
    parser.add_argument("--bootstrap-resamples", type=int, help="override policy.bootstrap_resamples")
    parser.add_argument(
        "--budget-measurements", type=int,
        help="stop racing once this many total measurement blocks have been spent across all candidates",
    )
    parser.add_argument(
        "--budget-seconds", type=int, help="stop racing once this many seconds have elapsed"
    )
    parser.add_argument(
        "--max-candidates", type=int,
        help="race at most this many pre-filtered candidates (kept in provider/warm-start order)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = JSONArgumentParser(
        prog="auto-mlx",
        description="Validate canonical Auto MLX documents, inspect their identities, and run the local "
        "evidence-gated evaluate/promote/dispatch loop -- without MLX at import time.",
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"auto-mlx {_package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=JSONArgumentParser)
    for command in ("validate", "inspect"):
        command_parser = subparsers.add_parser(
            command,
            help=(
                "validate a G0 contract or canonical document"
                if command == "validate"
                else "inspect canonical IDs without evaluating a candidate"
            ),
            description=(
                "Validate a G0 contract or canonical document."
                if command == "validate"
                else "Validate, then inspect the canonical IDs in a G0 contract or document."
            ),
            allow_abbrev=False,
        )
        _kind_arguments(command_parser, command=command)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="run a real local evaluation and store an attested receipt",
        description="Evaluate a candidate against its workload's baseline under the local sandbox tier, "
        "then store and attempt to attest the resulting receipt. Requires the local sandbox execution "
        "primitives (macOS, sandbox-exec on PATH); fails closed with exit code 4 otherwise.",
        allow_abbrev=False,
    )
    _context_document_arguments(evaluate_parser, policy_overrides=True, artifact_root_required=True)
    evaluate_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="run an A/A (candidate == baseline) calibration evaluation instead of a real candidate evaluation: "
        "forces policy.calibration=True and runs the baseline runner on both arms, so the resulting "
        "receipt's statistics measure this policy's true noise floor rather than a real candidate effect. "
        "Calibration receipts are structurally valid evidence but are never promotable (see auto_mlx.promotion).",
    )

    promote_parser = subparsers.add_parser(
        "promote",
        help="independently re-attest a stored receipt and decide activation",
        description="Load a stored receipt, independently re-verify and attest its evidence chain through "
        "the local supervisor, decide activation, and persist the resulting decision and pointer.",
        allow_abbrev=False,
    )
    promote_parser.add_argument("--receipt", required=True, dest="receipt_id", help="receipt_id to promote")
    promote_parser.add_argument("--store", help="receipt/decision store root (defaults to AUTO_MLX_STORE or ./auto-mlx-store)")
    promote_parser.add_argument(
        "--key-dir", help="attestation key directory (defaults to AUTO_MLX_KEY_DIR or ~/.auto-mlx/keys)"
    )
    promote_parser.add_argument(
        "--artifact-root",
        help="local root used to verify declared artifacts at activation time (defaults to the current directory)",
    )

    dispatch_parser = subparsers.add_parser(
        "dispatch",
        help="resolve (and optionally run) the currently active candidate or native fallback",
        description="Match the current activation decision against the given evaluation context and report "
        "candidate or native_fallback. With --execute, actually run the selected side under the local "
        "sandbox tier and report its output digest and duration.",
        allow_abbrev=False,
    )
    _context_document_arguments(dispatch_parser, policy_overrides=False, artifact_root_required=False)
    dispatch_parser.add_argument(
        "--execute", action="store_true", help="actually run the selected side under the local sandbox tier"
    )

    rollback_parser = subparsers.add_parser(
        "rollback",
        help="force dispatch back to native fallback",
        description="Write an immutable rollback decision and point dispatch at native code, regardless of "
        "the prior activation decision.",
        allow_abbrev=False,
    )
    rollback_parser.add_argument("--store", help="receipt/decision store root (defaults to AUTO_MLX_STORE or ./auto-mlx-store)")
    rollback_parser.add_argument(
        "--key-dir", help="attestation key directory (defaults to AUTO_MLX_KEY_DIR or ~/.auto-mlx/keys)"
    )

    tune_parser = subparsers.add_parser(
        "tune",
        help="race a provider's declarative knob grid against the baseline and store a tuning summary",
        description="Enumerate a declarative provider's candidate grid, pre-filter configs the workload's own "
        "knob contract excludes, race surviving candidates against the baseline under statistical elimination, "
        "and store a content-addressed tuning summary alongside a full attested receipt for every measured "
        "candidate. Requires the local sandbox execution primitives; fails closed with exit code 4 otherwise.",
        allow_abbrev=False,
    )
    _tune_context_arguments(tune_parser)

    history_parser = subparsers.add_parser(
        "history",
        help="list prior tuning summaries for a workload identity on this runtime",
        description="List content-addressed tuning summaries previously stored by `auto-mlx tune` for the "
        "exact (workload, runtime identity) pair; a mismatched workload or runtime finds no history.",
        allow_abbrev=False,
    )
    history_parser.add_argument("--workload", required=True, help="workload document path")
    history_parser.add_argument(
        "--runtime",
        help="runtime identity document path; if given, must match this host's current runtime identity "
        "(defaults to this host's current runtime identity)",
    )
    history_parser.add_argument(
        "--store", help="receipt/decision/tuning-summary store root (defaults to AUTO_MLX_STORE or ./auto-mlx-store)"
    )
    history_parser.add_argument(
        "--key-dir", help="attestation key directory (defaults to AUTO_MLX_KEY_DIR or ~/.auto-mlx/keys)"
    )

    keys_parser = subparsers.add_parser("keys", help="manage the local attestation key", allow_abbrev=False)
    keys_subparsers = keys_parser.add_subparsers(dest="keys_command", required=True, parser_class=JSONArgumentParser)
    keys_ensure_parser = keys_subparsers.add_parser(
        "ensure",
        help="create the local attestation key if missing, and report its path and fingerprint",
        allow_abbrev=False,
    )
    keys_ensure_parser.add_argument(
        "--key-dir", help="attestation key directory (defaults to AUTO_MLX_KEY_DIR or ~/.auto-mlx/keys)"
    )
    return parser


def _diagnostic(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        },
    }


def _write_diagnostic(value: dict[str, Any]) -> None:
    try:
        sys.stderr.write(canonical_json(value) + "\n")
        sys.stderr.flush()
    except (BrokenPipeError, OSError, UnicodeError):
        # There is no reliable destination left for a diagnostic. Never turn
        # a handled CLI failure into an interpreter traceback.
        return


def _quieten_broken_stdout() -> None:
    """Prevent interpreter shutdown from reporting a second broken pipe."""

    try:
        descriptor = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        return
    try:
        null_descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_descriptor, descriptor)
        finally:
            os.close(null_descriptor)
    except OSError:
        return


def _write_stdout(value: str) -> None:
    try:
        sys.stdout.write(value)
        sys.stdout.flush()
    except (BrokenPipeError, OSError) as exc:
        _quieten_broken_stdout()
        raise CLIIOError("cannot write command result to stdout") from exc


def _input_limit_error(source: str) -> AutoMLXError:
    return AutoMLXError(
        f"JSON input exceeds the {MAX_JSON_INPUT_BYTES} byte limit: {source}",
        code=FailureCode.INPUT_TOO_LARGE,
    )


def _read_bounded(stream: Any, *, source: str, text_stream: bool = False) -> bytes:
    """Read at most ``MAX_JSON_INPUT_BYTES`` before JSON parsing begins."""

    chunks: list[bytes] = []
    total = 0
    while total <= MAX_JSON_INPUT_BYTES:
        remaining = MAX_JSON_INPUT_BYTES - total
        # A text stream has no byte-counted read API. Four UTF-8 bytes per
        # character is conservative, and the encoded chunk is checked below.
        read_size = max(1, remaining // 4 + 1) if text_stream else remaining + 1
        chunk = stream.read(read_size)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        elif isinstance(chunk, bytearray):
            chunk = bytes(chunk)
        if not isinstance(chunk, bytes):
            raise AutoMLXError("JSON input stream did not return bytes or text", code=FailureCode.INVALID_JSON)
        total += len(chunk)
        if total > MAX_JSON_INPUT_BYTES:
            raise _input_limit_error(source)
        chunks.append(chunk)
    return b"".join(chunks)


def _read_bounded_descriptor(descriptor: int, *, source: str) -> bytes:
    """Read bounded input from the already-open descriptor."""

    chunks: list[bytes] = []
    total = 0
    while total <= MAX_JSON_INPUT_BYTES:
        chunk = os.read(descriptor, MAX_JSON_INPUT_BYTES - total + 1)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_JSON_INPUT_BYTES:
            raise _input_limit_error(source)
        chunks.append(chunk)
    return b"".join(chunks)


def _require_walk_primitives(path: str) -> None:
    if not callable(getattr(os, "open", None)) or not _OPEN_SUPPORTS_DIR_FD:
        raise CLIIOError(f"cannot safely access path {path}: descriptor-relative open is unavailable")
    if not callable(getattr(os, "fstat", None)):
        raise CLIIOError(f"cannot safely access path {path}: fstat is unavailable")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise CLIIOError(f"cannot safely access path {path}: no-follow directory opens are unavailable")


def _path_components(path_value: str, *, label: str) -> tuple[Path, str, tuple[str, ...]]:
    if type(path_value) is not str or not path_value:
        raise CLIIOError(f"{label} path must be a non-empty string")
    try:
        path = Path(path_value)
    except (TypeError, ValueError) as exc:
        raise CLIIOError(f"cannot parse {label} path {path_value!r}: {exc}") from exc
    parts = path.parts
    anchor = path.anchor or "."
    components = parts[1:] if path.anchor else parts
    if not components:
        raise CLIIOError(f"{label} path has no final name: {path_value!r}")
    return path, anchor, tuple(components)


def _walk_error(
    path: Path, component: str, exc: OSError | ValueError | NotImplementedError, *, descriptor: int | None = None
) -> CLIIOError:
    if not isinstance(exc, OSError):
        return CLIIOError(f"cannot open {path} parent component {component!r}: {exc}")
    if exc.errno == errno.ELOOP:
        return CLIIOError(f"{path} contains a symlink ancestor at {component!r}")
    if (
        exc.errno == errno.ENOTDIR
        and descriptor is not None
        and _STAT_SUPPORTS_DIR_FD
        and _STAT_SUPPORTS_NOFOLLOW
    ):
        try:
            entry = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
        except (OSError, ValueError, NotImplementedError):
            entry = None
        if entry is not None and stat.S_ISLNK(entry.st_mode):
            return CLIIOError(f"{path} contains a symlink ancestor at {component!r}")
    if exc.errno == errno.ENOENT:
        return CLIIOError(f"{path} has a missing parent component: {component!r}")
    if exc.errno in {errno.ENOTDIR, errno.EISDIR}:
        return CLIIOError(f"{path} has a non-directory parent component: {component!r}")
    return CLIIOError(f"cannot open {path} parent component {component!r}: {exc}")


def _close_owned(descriptors: list[int]) -> list[tuple[int, OSError | ValueError]]:
    """Attempt every owned close and discard ownership even when close fails."""

    failures: list[tuple[int, OSError | ValueError]] = []
    while descriptors:
        descriptor = descriptors.pop()
        try:
            os.close(descriptor)
        except (OSError, ValueError) as exc:
            failures.append((descriptor, exc))
    return failures


def _cleanup_detail(failures: list[tuple[int, OSError | ValueError]]) -> str:
    return "; ".join(f"fd {descriptor}: {error}" for descriptor, error in failures)


def _cli_cleanup_error(
    primary: Exception | None,
    failures: list[tuple[int, OSError | ValueError]],
    *,
    context: str,
) -> Exception:
    detail = _cleanup_detail(failures)
    if primary is None:
        return CLIIOError(f"{context} cleanup failed: {detail}")
    if isinstance(primary, CLIIOError):
        return CLIIOError(f"{primary}; {context} cleanup failed: {detail}")
    if isinstance(primary, AutoMLXError):
        return ContractError(f"{primary.message}; {context} cleanup failed: {detail}", code=primary.code)
    return CLIIOError(f"{primary}; {context} cleanup failed: {detail}")


def _open_parent_directory(path_value: str, *, label: str) -> tuple[int, str, Path]:
    """Open every parent component from a stable anchor without following links."""

    path, anchor, components = _path_components(path_value, label=label)
    _require_walk_primitives(str(path))
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    owned: list[int] = []
    try:
        descriptor = os.open(anchor, flags)
        owned.append(descriptor)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise CLIIOError(f"{label} anchor is not a directory: {anchor}")
        for component in components[:-1]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except (OSError, ValueError, NotImplementedError) as exc:
                raise _walk_error(path, component, exc, descriptor=descriptor) from exc
            owned.append(child)
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                raise CLIIOError(f"{path} has a non-directory parent component: {component!r}")
            try:
                os.close(descriptor)
            except (OSError, ValueError) as exc:
                owned.remove(descriptor)
                raise CLIIOError(f"cannot close {label} parent descriptor for {path}: {exc}") from exc
            owned.remove(descriptor)
            descriptor = child
        if descriptor is None:  # pragma: no cover - _path_components guarantees a component
            raise CLIIOError(f"cannot open {label} parent for {path}")
        return descriptor, components[-1], path
    except CLIIOError as primary:
        failures = _close_owned(owned)
        if failures:
            raise _cli_cleanup_error(primary, failures, context=f"{label} descriptor") from primary
        raise
    except (OSError, ValueError, NotImplementedError) as exc:
        primary = _walk_error(path, anchor, exc)
        failures = _close_owned(owned)
        if failures:
            raise _cli_cleanup_error(primary, failures, context=f"{label} descriptor") from primary
        raise primary from exc


def _read_json(path_value: str) -> Any:
    if path_value == "-":
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        raw = _read_bounded(stream, source="stdin", text_stream=stream is sys.stdin)
    else:
        parent_descriptor, name, path = _open_parent_directory(path_value, label="input")
        descriptor: int | None = None
        operation_error: Exception | None = None
        try:
            if not callable(getattr(os, "read", None)) or getattr(os, "O_NONBLOCK", None) is None:
                raise CLIIOError(f"cannot safely open input {path}: non-blocking descriptor reads are unavailable")
            try:
                entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            except (OSError, ValueError, NotImplementedError) as exc:
                if isinstance(exc, OSError) and exc.errno in _NON_REGULAR_OPEN_ERRNOS:
                    raise CLIIOError(f"input path is not a regular file: {path}") from exc
            else:
                if stat.S_ISLNK(entry.st_mode):
                    raise CLIIOError(f"input path contains a symlink: {path}")
                if not stat.S_ISREG(entry.st_mode):
                    raise CLIIOError(f"input path is not a regular file: {path}")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise CLIIOError(f"input path is not a regular file: {path}")
            if info.st_size > MAX_JSON_INPUT_BYTES:
                raise _input_limit_error(str(path))
            raw = _read_bounded_descriptor(descriptor, source=str(path))
        except (CLIIOError, AutoMLXError) as exc:
            operation_error = exc
        except (OSError, ValueError, NotImplementedError) as exc:
            if isinstance(exc, OSError) and exc.errno in _NON_REGULAR_OPEN_ERRNOS:
                operation_error = CLIIOError(f"input path is not a regular file: {path}")
            else:
                operation_error = CLIIOError(f"cannot read input {path}: {exc}")
        owned = [parent_descriptor]
        if descriptor is not None:
            owned.append(descriptor)
        close_errors = _close_owned(owned)
        if close_errors:
            combined = _cli_cleanup_error(operation_error, close_errors, context=f"input {path} descriptor")
            if operation_error is not None:
                raise combined from operation_error
            raise combined from close_errors[0][1]
        if operation_error is not None:
            raise operation_error
    return strict_json_loads(raw)


def _read_document_json(path_value: str, *, kind: str) -> Any:
    """Read JSON input for a document command.

    The schema-free ``document`` kind promises only that its input is valid
    canonical JSON, so every structural JSON failure it can hit -- too deep,
    an unpaired surrogate, a duplicate key -- is reported as the single
    stable ``invalid_json`` diagnostic, regardless of which specific rule
    caught it. Schema-bearing kinds (workload, receipt, ...) keep the
    precise failure code so callers can tell a bad field apart from
    unparseable input.
    """

    try:
        return _read_json(path_value)
    except CanonicalJSONError as exc:
        if kind != "document":
            raise
        raise CanonicalJSONError(str(exc)) from exc


def _resolve_path(args: argparse.Namespace) -> str:
    paths = [value for value in (args.path, args.input_option) if value is not None]
    if len(paths) != 1:
        raise CLIUsageError("provide exactly one document path as a positional argument or with --input")
    return paths[0]


def _resolve_kind(args: argparse.Namespace) -> str:
    kinds = [value for value in (args.kind, args.kind_option) if value is not None]
    if not kinds:
        raise CLIUsageError("provide a document kind, for example: validate workload --input workload.json")
    if len(kinds) == 2 and kinds[0] != kinds[1]:
        raise CLIUsageError("positional kind and --kind must agree")
    return kinds[0]


def _as_document(kind: str, value: Any, *, workload_value: Any | None, artifact_root: str | None) -> Any:
    if kind == "document":
        canonical_json(value)
        return value
    if kind == "receipt":
        receipt = Receipt.from_dict(value)
        validation = validate_receipt(receipt, artifact_root=artifact_root)
        if not validation.valid:
            detail = "; ".join(failure.message for failure in validation.failures)
            raise ContractError(f"receipt failed independent validation: {detail}", code=FailureCode.IDENTITY_MISMATCH)
        return receipt
    if kind == "artifact":
        return Artifact.from_dict(value)
    if kind == "knob":
        return Knob.from_dict(value)
    if kind == "policy":
        return EvaluationPolicy.from_dict(value)
    if kind == "provider":
        return DeclarativeProvider.from_dict(value)
    if kind == "runtime":
        return RuntimeIdentity.from_dict(value)
    if kind == "workload":
        workload = FrozenWorkload.from_dict(value)
        if artifact_root is not None:
            for artifact in workload.artifacts:
                artifact.verify(artifact_root)
        return workload
    if kind == "candidate":
        if workload_value is None:
            raise CLIUsageError("candidate validation requires --workload WORKLOAD.json")
        workload = FrozenWorkload.from_dict(workload_value)
        if artifact_root is not None:
            for artifact in workload.artifacts:
                artifact.verify(artifact_root)
        return CandidateProposal.from_dict(value, workload)
    if artifact_root is not None:
        raise CLIUsageError(f"--artifact-root is only valid for workload, candidate, or receipt documents")
    raise CLIUsageError(f"unsupported document kind: {kind}")


def _to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _identity_fields(kind: str, value: Any, document: Any) -> dict[str, Any]:
    if kind == "artifact":
        return {"sha256": document.sha256, "size_bytes": document.size_bytes}
    if kind == "knob":
        return {"knob_id": sha256_hex(document.to_dict())}
    if kind == "policy":
        return {"policy_id": sha256_hex(document.to_dict())}
    if kind == "provider":
        return {"provider_id": document.provider_id, "provider_hash": sha256_hex(document.to_dict())}
    if kind == "runtime":
        return {"runtime_id": document.identity}
    if kind == "workload":
        return {"workload_id": document.workload_id, "workload_hash": document.workload_hash}
    if kind == "candidate":
        return {
            "candidate_id": document.candidate_id,
            "provider_id": document.provider_id,
            "workload_hash": document.workload_hash,
        }
    if kind == "receipt":
        return {
            "receipt_id": document.receipt_id,
            "candidate_id": document.candidate.candidate_id,
            "workload_hash": document.workload.workload_hash,
            "status": document.status,
        }
    return {"document_id": sha256_hex(value)}


def _write_descriptor(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("output descriptor made no progress")
        view = view[written:]


def _require_publication_primitives(path: str) -> None:
    _require_walk_primitives(path)
    if not _STAT_SUPPORTS_DIR_FD or not _STAT_SUPPORTS_NOFOLLOW:
        raise CLIIOError(f"cannot safely publish output {path}: descriptor-relative no-follow stat is unavailable")
    if not _LINK_SUPPORTS_DIR_FD or not _LINK_SUPPORTS_NOFOLLOW or not callable(getattr(os, "link", None)):
        raise CLIIOError(f"cannot safely publish output {path}: no-follow hard-link publication is unavailable")
    if not _UNLINK_SUPPORTS_DIR_FD or not callable(getattr(os, "unlink", None)):
        raise CLIIOError(f"cannot safely publish output {path}: descriptor-relative unlink is unavailable")
    if not callable(getattr(os, "fchmod", None)):
        raise CLIIOError(f"cannot safely publish output {path}: fchmod is unavailable")
    if not callable(getattr(os, "fsync", None)):
        raise CLIIOError(f"cannot safely publish output {path}: fsync is unavailable")
    if not callable(getattr(os, "write", None)) or not callable(getattr(os, "fstat", None)):
        raise CLIIOError(f"cannot safely publish output {path}: descriptor writes are unavailable")


def _identity_from_stat(value: os.stat_result, *, path: str) -> tuple[int, int]:
    try:
        return value.st_dev, value.st_ino
    except AttributeError as exc:
        raise CLIIOError(f"cannot safely publish output {path}: device/inode metadata is unavailable") from exc


def _descriptor_identity(descriptor: int, *, path: str) -> tuple[int, int]:
    try:
        return _identity_from_stat(os.fstat(descriptor), path=path)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise CLIIOError(f"cannot inspect open output descriptor for {path}: {exc}") from exc


def _name_identity(descriptor: int, name: str, *, path: str) -> tuple[int, int]:
    try:
        return _identity_from_stat(os.stat(name, dir_fd=descriptor, follow_symlinks=False), path=path)
    except (OSError, ValueError, NotImplementedError) as exc:
        raise CLIIOError(f"cannot inspect output name {path}: {exc}") from exc


def _unlink_if_identity(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int],
    *,
    path: str,
) -> tuple[bool, str]:
    """Remove a name only after re-checking that it is still our inode."""

    last_error: OSError | None = None
    for _ in range(2):
        try:
            current = _name_identity(directory_descriptor, name, path=path)
        except CLIIOError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return True, "already absent"
            last_error = exc.__cause__ if isinstance(exc.__cause__, OSError) else OSError(str(exc))
            continue
        if current != identity:
            return False, "identity changed; the name was left untouched"
        try:
            os.unlink(name, dir_fd=directory_descriptor)
            return True, "removed"
        except FileNotFoundError:
            return True, "already absent"
        except (OSError, ValueError, NotImplementedError) as exc:
            last_error = exc
    return False, f"could not remove the identity-matched name: {last_error}"


def _output_trailing_separator(path_value: str) -> bool:
    separators = {os.sep}
    if os.altsep is not None:
        separators.add(os.altsep)
    return any(path_value.endswith(separator) for separator in separators)


def _create_only(path_value: str, payload: bytes) -> None:
    """Publish a complete private staged inode without replacing a name."""

    if type(path_value) is not str or not path_value:
        raise CLIIOError("output path must be a non-empty string")
    if _output_trailing_separator(path_value):
        raise CLIIOError(f"output path must name a file and must not end in a separator: {path_value!r}")
    path, _, components = _path_components(path_value, label="output")
    name = components[-1]
    if name in {".", ".."}:
        raise CLIIOError(f"output path has an invalid final name: {path_value!r}")
    _require_publication_primitives(str(path))
    directory_descriptor, _, _ = _open_parent_directory(path_value, label="output")
    staged_descriptor: int | None = None
    temporary_name: str | None = None
    staged_identity: tuple[int, int] | None = None
    published = False
    rollback_required = False
    operation_error: CLIIOError | None = None
    close_errors: list[tuple[int, OSError | ValueError]] = []
    cleanup_messages: list[str] = []

    try:
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            open_flags |= os.O_CLOEXEC
        for _ in range(32):
            temporary_name = f".{name}.auto-mlx-{secrets.token_hex(16)}"
            try:
                staged_descriptor = os.open(
                    temporary_name,
                    open_flags,
                    0o600,
                    dir_fd=directory_descriptor,
                )
                break
            except FileExistsError:
                temporary_name = None
        if staged_descriptor is None or temporary_name is None:
            raise CLIIOError(f"could not create an unpredictable private staging file in {path.parent}")

        staged_identity = _descriptor_identity(staged_descriptor, path=str(path))
        if not stat.S_ISREG(os.fstat(staged_descriptor).st_mode):
            raise CLIIOError(f"private staging file is not regular: {path}")
        try:
            _write_descriptor(staged_descriptor, payload)
        except (OSError, ValueError) as exc:
            raise CLIIOError(f"staged output contents were not written; no final was published: {path}: {exc}") from exc
        try:
            os.fchmod(staged_descriptor, 0o600)
        except (OSError, ValueError) as exc:
            raise CLIIOError(
                f"staged output private permissions were not confirmed; no final was published: {path}: {exc}"
            ) from exc
        if stat.S_IMODE(os.fstat(staged_descriptor).st_mode) != 0o600:
            raise CLIIOError(f"private staging permissions could not be confirmed: {path}")
        try:
            os.fsync(staged_descriptor)
        except (OSError, ValueError) as exc:
            raise CLIIOError(
                f"staged output contents were not durably synced; no final was published: {path}: {exc}"
            ) from exc
        if _name_identity(directory_descriptor, temporary_name, path=str(path)) != staged_identity:
            raise CLIIOError(f"private staging name changed before publication: {path}")

        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError as exc:
            raise CLIIOError(f"refusing to overwrite existing output: {path}") from exc
        except (NotImplementedError, OSError, TypeError, ValueError) as exc:
            # A syscall can fail after creating the link. Only treat it as a
            # publication when the destination now names this open inode.
            try:
                destination_identity = _name_identity(directory_descriptor, name, path=str(path))
            except CLIIOError as identity_error:
                raise CLIIOError(
                    f"cannot publish output {path}: {exc}; destination identity could not be checked: {identity_error}"
                ) from exc
            if destination_identity == staged_identity:
                published = True
            else:
                raise CLIIOError(f"cannot publish output {path}: {exc}") from exc
        if not published:
            raise CLIIOError(f"cannot publish output {path}: destination was not created")

        rollback_required = True
        if _name_identity(directory_descriptor, name, path=str(path)) != staged_identity:
            raise CLIIOError(f"published output identity did not match the staged descriptor: {path}")
        if _name_identity(directory_descriptor, name, path=str(path)) != staged_identity:
            raise CLIIOError(f"published output identity changed during verification: {path}")
        try:
            os.fsync(directory_descriptor)
        except (OSError, ValueError) as exc:
            raise CLIIOError(
                f"post-link directory durability is uncertain; attempting rollback: {path}: {exc}"
            ) from exc

        try:
            removed, detail = _unlink_if_identity(
                directory_descriptor,
                temporary_name,
                staged_identity,
                path=str(path),
            )
        except CLIIOError as exc:
            raise CLIIOError(f"private staging cleanup could not be checked: {path}: {exc}") from exc
        if not removed:
            raise CLIIOError(f"private staging cleanup failed for {path}: {detail}")
        cleanup_messages.append(f"private staging {detail}")
        temporary_name = None
        try:
            os.fsync(directory_descriptor)
        except (OSError, ValueError) as exc:
            raise CLIIOError(
                f"private staging removal durability is uncertain; attempting rollback: {path}: {exc}"
            ) from exc
        rollback_required = False
    except CLIIOError as exc:
        operation_error = exc
    except (OSError, TypeError, ValueError, NotImplementedError) as exc:
        operation_error = CLIIOError(f"cannot publish output {path}: {exc}")
    finally:
        if operation_error is not None and published and rollback_required and staged_identity is not None:
            removed, detail = _unlink_if_identity(
                directory_descriptor,
                name,
                staged_identity,
                path=str(path),
            )
            cleanup_messages.append(f"destination rollback: {detail}")
            if removed:
                try:
                    os.fsync(directory_descriptor)
                except (OSError, ValueError, NotImplementedError) as exc:
                    cleanup_messages.append(f"destination rollback durability is uncertain: {exc}")
        if temporary_name is not None and staged_identity is not None:
            removed, detail = _unlink_if_identity(
                directory_descriptor,
                temporary_name,
                staged_identity,
                path=str(path),
            )
            cleanup_messages.append(f"private staging cleanup: {detail}")
            if removed:
                temporary_name = None
        owned = [directory_descriptor]
        if staged_descriptor is not None:
            owned.append(staged_descriptor)
        close_errors.extend(_close_owned(owned))

    if close_errors:
        close_detail = _cleanup_detail(close_errors)
        if operation_error is None:
            operation_error = CLIIOError(
                f"output was published, but descriptor close failed; inspect {path}: {close_detail}"
            )
        else:
            operation_error = CLIIOError(f"{operation_error}; descriptor close failed: {close_detail}")
    if operation_error is not None:
        if cleanup_messages:
            operation_error = CLIIOError(f"{operation_error} ({'; '.join(cleanup_messages)})")
        raise operation_error


def _run_document_command(args: argparse.Namespace) -> dict[str, Any]:
    kind = _resolve_kind(args)
    if args.workload is not None and kind != "candidate":
        raise CLIUsageError("--workload is only valid for candidate validation")
    if args.artifact_root is not None and kind not in {"candidate", "receipt", "workload"}:
        raise CLIUsageError("--artifact-root is only valid for workload, candidate, or receipt documents")
    input_path = _resolve_path(args)
    value = _read_document_json(input_path, kind=kind)
    workload_value = _read_document_json(args.workload, kind=kind) if args.workload is not None else None
    document = _as_document(kind, value, workload_value=workload_value, artifact_root=args.artifact_root)
    document_value = _to_dict(document)
    canonical = canonical_json(document_value)
    output_path = getattr(args, "output", None)
    if output_path is not None:
        _create_only(output_path, canonical.encode("utf-8"))
    result: dict[str, Any] = {
        "ok": True,
        "command": args.command,
        "kind": kind,
        "canonical_sha256": sha256_hex(document_value),
    }
    if args.command == "validate":
        result["document"] = document_value
    else:
        result["ids"] = _identity_fields(kind, value, document)
    if output_path is not None:
        result["output"] = output_path
    return result


def _apply_policy_overrides(policy: EvaluationPolicy, args: argparse.Namespace) -> EvaluationPolicy:
    overrides: dict[str, Any] = {}
    for attribute, field_name in (
        ("measurement_runs", "measurement_runs"),
        ("warmup_runs", "warmup_runs"),
        ("timeout_seconds", "timeout_seconds"),
        ("max_output_bytes", "max_output_bytes"),
        ("k_repetitions", "k_repetitions"),
        ("max_measurement_runs", "max_measurement_runs"),
        ("min_effect_bps", "min_effect_bps"),
        ("bootstrap_resamples", "bootstrap_resamples"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[field_name] = value
    # --calibrate is the single source of truth for policy.calibration: it
    # is equivalent to supplying a --policy document with calibration=true,
    # and _run_evaluate_command reads policy.calibration (never args.calibrate
    # directly) to decide whether to force the candidate arm onto the
    # baseline runner -- so the two entry points stay self-consistent.
    if getattr(args, "calibrate", False):
        overrides["calibration"] = True
    if not overrides:
        return policy
    fields = policy.to_dict()
    fields.update(overrides)
    return EvaluationPolicy.from_dict(fields)


def _load_evaluation_context(
    args: argparse.Namespace,
) -> tuple[FrozenWorkload, CandidateProposal, EvaluationPolicy, RuntimeIdentity]:
    """Load workload/candidate/policy/runtime through the same contract layer ``validate`` uses.

    ``--workload`` and ``--candidate`` are always read and validated;
    ``--policy`` and ``--runtime`` default (evaluation-policy defaults and
    this host's current runtime identity) when omitted.  A given
    ``--runtime`` document must match this host's actual current runtime
    identity -- it is never used to impersonate a different host.
    """

    artifact_root = args.artifact_root
    workload_value = _read_document_json(args.workload, kind="workload")
    workload = _as_document("workload", workload_value, workload_value=None, artifact_root=artifact_root)
    candidate_value = _read_document_json(args.candidate, kind="candidate")
    candidate = _as_document("candidate", candidate_value, workload_value=workload_value, artifact_root=artifact_root)
    if args.policy is not None:
        policy_value = _read_document_json(args.policy, kind="policy")
        policy = _as_document("policy", policy_value, workload_value=None, artifact_root=None)
    else:
        policy = EvaluationPolicy()
    policy = _apply_policy_overrides(policy, args)
    runtime = _resolve_current_runtime(args)
    return workload, candidate, policy, runtime


def _require_local_sandbox(*, stage: str) -> None:
    if not local_sandbox_primitives_available():
        raise CLIUnavailableError(
            "this command requires the local sandbox execution primitives (macOS, sandbox-exec on PATH, "
            "and descriptor-relative artifact access), which are not available on this host",
            stage=stage,
            surface="local_sandbox",
        )


def _run_evaluate_command(args: argparse.Namespace) -> dict[str, Any]:
    workload, candidate, policy, _runtime = _load_evaluation_context(args)
    artifact_root = args.artifact_root

    # Host-independent: an unrecognized workload is a typed diagnostic on
    # every host, never gated behind sandbox availability.
    registry = TrustedRunnerRegistry()
    baseline_runner_id, candidate_runner_id = _resolve_workload_runners(workload, registry)
    if policy.calibration:
        # A/A calibration: both arms run the identical baseline runner, so
        # the resulting receipt's statistics measure this policy's true
        # noise floor rather than any real candidate effect.  policy.calibration
        # is the single source of truth for this (see _apply_policy_overrides);
        # --calibrate is just one way to set it.
        candidate_runner_id = baseline_runner_id

    _require_local_sandbox(stage="G1")

    execution_policy = _execution_policy_from_evaluation_policy(policy)

    # The source-of-truth oracle comes from one real baseline execution, not
    # a hardcoded literal -- matching tests/test_supervisor.py's E2E chain.
    probe_plan = build_execution_plan(candidate, registry, baseline_runner_id, artifact_root)
    probe_record = probe_plan.execute(
        execution_policy,
        registry=registry,
        provider=LocalSandboxProvider(),
        authority=LocalSandboxAuthority(),
    )
    if probe_record.status is not ExecutionStatus.SUCCESS:
        detail = probe_record.failure.message if probe_record.failure is not None else probe_record.status.value
        raise ContractError(
            f"baseline probe execution did not succeed: {detail}",
            code=FailureCode.RUNTIME_FAILURE,
        )
    oracle = ExactOutputOracle(probe_record.stdout)

    evaluator = Evaluator(
        registry,
        baseline_runner_id=baseline_runner_id,
        candidate_runner_id=candidate_runner_id,
        oracle=oracle,
        artifact_root=artifact_root,
        policy=policy,
        execution_policy=execution_policy,
        provider=LocalSandboxProvider(),
        authority=LocalSandboxAuthority(),
    )
    bundle = evaluator.evaluate(candidate)
    receipt = Receipt.from_observation_bundle(
        bundle, workload, candidate, policy, oracle=oracle, created_at_ns=time.time_ns()
    )

    store = store_config.open_store(args.store, key_dir=args.key_dir)
    store.put_receipt(receipt, require_durable=True)

    # Auto-ensure the local attestation key on first use, then attempt
    # supervisor attestation -- never key material in any returned value.
    key = keys_module.ensure_attestation_key(key_dir=args.key_dir)
    attested = False
    attestation_refusal: str | None = None
    try:
        attest_receipt(receipt, key, artifact_root=artifact_root)
        attested = True
    except SupervisorRefusalError as exc:
        attestation_refusal = str(exc)

    receipt_wire = receipt.to_dict()
    result: dict[str, Any] = {
        "ok": True,
        "command": "evaluate",
        "receipt_id": receipt.receipt_id,
        "status": receipt.status,
        "attested": attested,
        "store": str(store.root),
        "candidate_id": candidate.candidate_id,
        "workload_hash": workload.workload_hash,
        "workload_name": workload.name,
        "baseline_runner_id": baseline_runner_id,
        "candidate_runner_id": candidate_runner_id,
        "isolation_tier": bundle.isolation_provider_id,
        "gain": receipt_wire["metrics"]["gain"],
        "statistics": receipt_wire.get("statistics"),
        "calibration": policy.calibration,
    }
    if attestation_refusal is not None:
        result["attestation_refusal"] = attestation_refusal
    return result


def _run_promote_command(args: argparse.Namespace) -> dict[str, Any]:
    store = store_config.open_store(args.store, key_dir=args.key_dir)
    receipt = store.get_receipt(args.receipt_id)
    # Strict, non-generating load: promote never silently mints a fresh key
    # for a receipt it did not itself just evaluate.
    key = keys_module.load_attestation_key(key_dir=args.key_dir)
    artifact_root = args.artifact_root if args.artifact_root is not None else str(Path.cwd())

    attestation: str | None = None
    attestation_refusal: str | None = None
    try:
        attestation = attest_receipt(receipt, key, artifact_root=artifact_root)
    except SupervisorRefusalError as exc:
        attestation_refusal = str(exc)

    validation = validate_receipt(receipt, artifact_root=artifact_root, attestation=attestation, attestation_key=key)
    activated = activate_decision(store, validation, artifact_root=artifact_root, attestation_key=key, now_ns=time.time_ns())

    gain = validation.recomputed.get("metrics", {}).get("gain", {}) if validation.recomputed else {}
    statistics = validation.recomputed.get("statistics") if validation.recomputed else None
    result: dict[str, Any] = {
        "ok": True,
        "command": "promote",
        "receipt_id": receipt.receipt_id,
        "action": activated.action,
        "reason": activated.reason,
        "decision_id": activated.decision_id,
        "attested": attestation is not None,
        "current_decision_id": store.current_decision_id(),
        "store": str(store.root),
        "gain": gain,
        "statistics": statistics,
    }
    if attestation_refusal is not None:
        result["attestation_refusal"] = attestation_refusal
    return result


def _run_dispatch_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_root is None:
        args.artifact_root = str(Path.cwd())
    workload, candidate, policy, runtime = _load_evaluation_context(args)
    artifact_root = args.artifact_root

    store = store_config.open_store(args.store, key_dir=args.key_dir)
    try:
        key: bytes | None = keys_module.load_attestation_key(key_dir=args.key_dir)
    except KeyMaterialError:
        # Dispatch is a safety boundary: a missing/invalid local key means
        # activation can never be independently verified, so this degrades
        # to native fallback rather than crashing the CLI command.
        key = None

    result = run_dispatch(
        store,
        workload,
        candidate,
        policy,
        runtime,
        artifact_root=artifact_root,
        attestation_key=key,
        now_ns=time.time_ns(),
        max_age_ns=DEFAULT_MAX_AGE_NS,
    )
    output: dict[str, Any] = {
        "ok": True,
        "command": "dispatch",
        "dispatch": result.to_dict(),
        "store": str(store.root),
    }
    if args.execute:
        _require_local_sandbox(stage="G2")
        registry = TrustedRunnerRegistry()
        baseline_runner_id, candidate_runner_id = _resolve_workload_runners(workload, registry)
        runner_id = candidate_runner_id if result.mode == CANDIDATE_MODE else baseline_runner_id
        execution_policy = _execution_policy_from_evaluation_policy(policy)
        plan = build_execution_plan(candidate, registry, runner_id, artifact_root)
        record = plan.execute(
            execution_policy,
            registry=registry,
            provider=LocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        if record.status is not ExecutionStatus.SUCCESS:
            detail = record.failure.message if record.failure is not None else record.status.value
            raise ContractError(f"dispatch --execute run did not succeed: {detail}", code=FailureCode.RUNTIME_FAILURE)
        output["execution"] = {
            "mode": result.mode,
            "runner_id": runner_id,
            "digest": hashlib.sha256(record.stdout).hexdigest(),
            "duration_ns": record.parent_elapsed_ns,
        }
    return output


def _load_tune_context(
    args: argparse.Namespace,
) -> tuple[FrozenWorkload, DeclarativeProvider, EvaluationPolicy, RuntimeIdentity]:
    artifact_root = args.artifact_root
    workload_value = _read_document_json(args.workload, kind="workload")
    workload = _as_document("workload", workload_value, workload_value=None, artifact_root=artifact_root)
    provider_value = _read_document_json(args.provider, kind="provider")
    provider = _as_document("provider", provider_value, workload_value=None, artifact_root=None)
    if args.policy is not None:
        policy_value = _read_document_json(args.policy, kind="policy")
        policy = _as_document("policy", policy_value, workload_value=None, artifact_root=None)
    else:
        policy = EvaluationPolicy()
    policy = _apply_policy_overrides(policy, args)
    runtime = _resolve_current_runtime(args)
    return workload, provider, policy, runtime


def _resolve_current_runtime(args: argparse.Namespace) -> RuntimeIdentity:
    if args.runtime is not None:
        runtime_value = _read_document_json(args.runtime, kind="runtime")
        runtime = _as_document("runtime", runtime_value, workload_value=None, artifact_root=None)
        current = RuntimeIdentity.current()
        if runtime.identity != current.identity:
            raise ContractError(
                "--runtime document does not match this host's current runtime identity",
                code=FailureCode.IDENTITY_MISMATCH,
            )
        return runtime
    return RuntimeIdentity.current()


def _run_tune_command(args: argparse.Namespace) -> dict[str, Any]:
    workload, provider, policy, runtime = _load_tune_context(args)
    artifact_root = args.artifact_root

    registry = TrustedRunnerRegistry()
    baseline_runner_id, candidate_runner_id = _resolve_workload_runners(workload, registry)

    _require_local_sandbox(stage="G1")

    legal, pruned = tune_module.prefilter_candidates(provider, workload)
    considered = len(provider.configs)
    ordered_capped, max_candidates_dropped = tune_module.apply_max_candidates(legal, args.max_candidates)

    store = store_config.open_store(args.store, key_dir=args.key_dir)
    ordered = tune_module.warm_start_order(
        ordered_capped, store=store, workload_hash=workload.workload_hash, runtime_identity=runtime.identity
    )

    if ordered:
        probe_execution_policy = _execution_policy_from_evaluation_policy(policy)
        probe_plan = build_execution_plan(ordered[0], registry, baseline_runner_id, artifact_root)
        probe_record = probe_plan.execute(
            probe_execution_policy,
            registry=registry,
            provider=LocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        if probe_record.status is not ExecutionStatus.SUCCESS:
            detail = probe_record.failure.message if probe_record.failure is not None else probe_record.status.value
            raise ContractError(
                f"baseline probe execution did not succeed: {detail}",
                code=FailureCode.RUNTIME_FAILURE,
            )
        oracle = ExactOutputOracle(probe_record.stdout)
        key = keys_module.ensure_attestation_key(key_dir=args.key_dir)

        def run_rung(candidate: CandidateProposal, rung_policy: EvaluationPolicy):
            rung_execution_policy = _execution_policy_from_evaluation_policy(rung_policy)
            evaluator = Evaluator(
                registry,
                baseline_runner_id=baseline_runner_id,
                candidate_runner_id=candidate_runner_id,
                oracle=oracle,
                artifact_root=artifact_root,
                policy=rung_policy,
                execution_policy=rung_execution_policy,
                provider=LocalSandboxProvider(),
                authority=LocalSandboxAuthority(),
            )
            return evaluator.evaluate(candidate)

        def store_rung_receipt(bundle: Any, candidate: CandidateProposal, rung_policy: EvaluationPolicy) -> tuple[str, bool]:
            receipt = Receipt.from_observation_bundle(
                bundle, workload, candidate, rung_policy, oracle=oracle, created_at_ns=time.time_ns()
            )
            store.put_receipt(receipt, require_durable=True)
            attested = False
            try:
                attest_receipt(receipt, key, artifact_root=artifact_root)
                attested = True
            except SupervisorRefusalError:
                pass
            return receipt.receipt_id, attested

        outcome = tune_module.race_candidates(
            candidates=ordered,
            base_policy=policy,
            run_rung=run_rung,
            store_receipt=store_rung_receipt,
            budget_measurements=args.budget_measurements,
            budget_seconds=args.budget_seconds,
        )
    else:
        outcome = tune_module.RaceOutcome(
            entrants=(),
            incumbent=None,
            blocks_spent=0,
            seconds_spent_ns=0,
            budget_measurements=args.budget_measurements,
            budget_seconds=args.budget_seconds,
            budget_exhausted=False,
        )

    summary = tune_module.build_tuning_summary(
        workload_hash=workload.workload_hash,
        runtime=runtime,
        provider_id=provider.provider_id,
        base_policy=policy,
        considered=considered,
        pruned=pruned,
        max_candidates=args.max_candidates,
        max_candidates_dropped=max_candidates_dropped,
        outcome=outcome,
    )
    store.put_tuning_summary(summary, require_durable=True)
    store.append_tuning_history(workload.workload_hash, runtime.identity, summary.summary_id, require_durable=True)

    result = summary.to_dict()
    result["ok"] = True
    result["command"] = "tune"
    result["store"] = str(store.root)
    return result


def _run_history_command(args: argparse.Namespace) -> dict[str, Any]:
    workload_value = _read_document_json(args.workload, kind="workload")
    workload = _as_document("workload", workload_value, workload_value=None, artifact_root=None)
    runtime = _resolve_current_runtime(args)
    store = store_config.open_store(args.store, key_dir=args.key_dir)
    summary_ids = store.list_tuning_history(workload.workload_hash, runtime.identity)
    summaries = [store.get_tuning_summary(summary_id) for summary_id in summary_ids]
    return {
        "ok": True,
        "command": "history",
        "workload_hash": workload.workload_hash,
        "runtime_id": runtime.identity,
        "store": str(store.root),
        "summaries": summaries,
    }


def _run_rollback_command(args: argparse.Namespace) -> dict[str, Any]:
    store = store_config.open_store(args.store, key_dir=args.key_dir)
    decision = rollback_decision(store, now_ns=time.time_ns())
    return {
        "ok": True,
        "command": "rollback",
        "action": decision.action,
        "reason": decision.reason,
        "decision_id": decision.decision_id,
        "current_decision_id": store.current_decision_id(),
        "store": str(store.root),
    }


def _run_keys_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.keys_command != "ensure":  # pragma: no cover - argparse restricts to known subcommands
        raise CLIUsageError(f"unknown keys subcommand: {args.keys_command}")
    resolved_dir = keys_module.resolve_key_dir(args.key_dir)
    key = keys_module.ensure_attestation_key(key_dir=args.key_dir)
    fingerprint = hashlib.sha256(key).hexdigest()[:16]
    return {
        "ok": True,
        "command": "keys",
        "subcommand": "ensure",
        "key_dir": str(resolved_dir),
        "key_path": str(resolved_dir / keys_module.KEY_FILE_NAME),
        "fingerprint_sha256_16": fingerprint,
    }


def _exit_for_error(error: AutoMLXError) -> int:
    return EXIT_CONTRACT


_COMMAND_HANDLERS: Final[dict[str, Callable[[argparse.Namespace], dict[str, Any]]]] = {
    "validate": _run_document_command,
    "inspect": _run_document_command,
    "evaluate": _run_evaluate_command,
    "promote": _run_promote_command,
    "dispatch": _run_dispatch_command,
    "tune": _run_tune_command,
    "history": _run_history_command,
    "rollback": _run_rollback_command,
    "keys": _run_keys_command,
}


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        handler = _COMMAND_HANDLERS.get(args.command)
        if handler is None:  # pragma: no cover - argparse restricts to known commands
            raise CLIUsageError(f"unknown command: {args.command}")
        result = handler(args)
        _write_stdout(canonical_json(result) + "\n")
        return EXIT_OK
    except SystemExit as exc:
        # argparse uses SystemExit for normal --help/--version output. Keep
        # the public ``main`` helper return-oriented for embedding and tests.
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    except CLIUsageError as exc:
        _write_diagnostic(_diagnostic("usage_error", str(exc)))
        return EXIT_USAGE
    except CLIIOError as exc:
        _write_diagnostic(_diagnostic("io_error", str(exc)))
        return EXIT_IO
    except CLIUnavailableError as exc:
        _write_diagnostic(
            _diagnostic(
                "unavailable",
                str(exc),
                details={"status": "unavailable", "stage": exc.stage, "surface": exc.surface},
            )
        )
        return EXIT_UNAVAILABLE
    except BrokenPipeError as exc:
        _quieten_broken_stdout()
        _write_diagnostic(_diagnostic("io_error", "cannot write command output"))
        return EXIT_IO
    except AutoMLXError as exc:
        _write_diagnostic(_diagnostic(exc.code.value, exc.message))
        return _exit_for_error(exc)
    except OSError as exc:
        _write_diagnostic(_diagnostic("io_error", str(exc)))
        return EXIT_IO
    except NotImplementedError as exc:
        _write_diagnostic(_diagnostic("io_error", str(exc)))
        return EXIT_IO
    except (RecursionError, UnicodeError) as exc:
        _write_diagnostic(_diagnostic("invalid_json", f"invalid JSON document: {exc}"))
        return EXIT_CONTRACT
    except Exception as exc:  # pragma: no cover - a final stable boundary for CLI callers
        _write_diagnostic(_diagnostic("internal_error", str(exc)))
        return EXIT_INTERNAL


__all__ = [
    "EXIT_CONTRACT",
    "EXIT_INTERNAL",
    "EXIT_IO",
    "EXIT_OK",
    "EXIT_UNAVAILABLE",
    "EXIT_USAGE",
    "MAX_JSON_INPUT_BYTES",
    "build_parser",
    "main",
]
