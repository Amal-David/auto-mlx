"""Dry-run-first, MLX-free command line interface for Auto MLX G0.

The CLI validates declarative documents and inspects identities.  Evaluator,
receipt, promotion, and dispatch libraries are available separately; their
CLI orchestration remains explicitly deferred until the required evidence and
activation boundaries are proven.
"""

from __future__ import annotations

import argparse
import errno
from importlib.metadata import PackageNotFoundError, version as distribution_version
import os
import secrets
import stat
import sys
import tomllib
from pathlib import Path
from typing import Any, Final

from .canonical import canonical_json, sha256_hex, strict_json_loads
from .contracts import (
    Artifact,
    CandidateProposal,
    EvaluationPolicy,
    FrozenWorkload,
    Knob,
    RuntimeIdentity,
)
from .errors import AutoMLXError, ContractError, FailureCode
from .providers import DeclarativeProvider
from .receipts import Receipt, validate_receipt


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
_DEFERRED_COMMANDS: Final = {
    "evaluate": (
        "CLI orchestration is deferred; the evaluator library exists, but production evaluation may fail closed "
        "until real isolation and supervisor proof are provided"
    ),
    "promote": (
        "CLI orchestration is deferred; the promotion library exists, but activation remains gated on later "
        "evidence and activation proof"
    ),
    "dispatch": (
        "CLI orchestration is deferred; the dispatch library exists, but activation remains gated on later "
        "evidence and activation proof"
    ),
}
_DEFERRED_STAGES: Final = {"evaluate": "G1", "promote": "G2", "dispatch": "G2"}


class CLIUsageError(Exception):
    """An argument error that can be rendered as the CLI's JSON diagnostic."""


class CLIIOError(Exception):
    """An input/output failure at the CLI boundary."""


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


def build_parser() -> argparse.ArgumentParser:
    parser = JSONArgumentParser(
        prog="auto-mlx",
        description="Validate canonical Auto MLX documents and inspect their identities without MLX.",
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
    for command, reason in _DEFERRED_COMMANDS.items():
        subparsers.add_parser(command, help=f"deferred: {reason}", allow_abbrev=False)
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
        except OSError:
            entry = None
        if entry is not None and stat.S_ISLNK(entry.st_mode):
            return CLIIOError(f"{path} contains a symlink ancestor at {component!r}")
    if exc.errno == errno.ENOENT:
        return CLIIOError(f"{path} has a missing parent component: {component!r}")
    if exc.errno in {errno.ENOTDIR, errno.EISDIR}:
        return CLIIOError(f"{path} has a non-directory parent component: {component!r}")
    return CLIIOError(f"cannot open {path} parent component {component!r}: {exc}")


def _open_parent_directory(path_value: str, *, label: str) -> tuple[int, str, Path]:
    """Open every parent component from a stable anchor without following links."""

    path, anchor, components = _path_components(path_value, label=label)
    _require_walk_primitives(str(path))
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(anchor, flags)
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise CLIIOError(f"{label} anchor is not a directory: {anchor}")
        for component in components[:-1]:
            child: int | None = None
            try:
                child = os.open(component, flags, dir_fd=descriptor)
                child_stat = os.fstat(child)
                if not stat.S_ISDIR(child_stat.st_mode):
                    raise CLIIOError(f"{path} has a non-directory parent component: {component!r}")
            except CLIIOError:
                if child is not None:
                    try:
                        os.close(child)
                    except OSError:
                        pass
                raise
            except (OSError, ValueError, NotImplementedError) as exc:
                if child is not None:
                    try:
                        os.close(child)
                    except OSError:
                        pass
                raise _walk_error(path, component, exc, descriptor=descriptor) from exc
            try:
                os.close(descriptor)
            except OSError as exc:
                try:
                    os.close(child)
                except OSError:
                    pass
                raise CLIIOError(f"cannot close {label} parent descriptor for {path}: {exc}") from exc
            descriptor = child
        if descriptor is None:  # pragma: no cover - _path_components guarantees a component
            raise CLIIOError(f"cannot open {label} parent for {path}")
        return descriptor, components[-1], path
    except CLIIOError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise
    except (OSError, ValueError, NotImplementedError) as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise _walk_error(path, anchor, exc) from exc


def _read_json(path_value: str) -> Any:
    if path_value == "-":
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        raw = _read_bounded(stream, source="stdin", text_stream=stream is sys.stdin)
    else:
        parent_descriptor, name, path = _open_parent_directory(path_value, label="input")
        descriptor: int | None = None
        operation_error: CLIIOError | None = None
        close_error: OSError | None = None
        try:
            if not callable(getattr(os, "read", None)) or getattr(os, "O_NONBLOCK", None) is None:
                raise CLIIOError(f"cannot safely open input {path}: non-blocking descriptor reads are unavailable")
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
        except CLIIOError as exc:
            operation_error = exc
        except (OSError, ValueError, NotImplementedError) as exc:
            operation_error = CLIIOError(f"cannot read input {path}: {exc}")
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    close_error = exc
            try:
                os.close(parent_descriptor)
            except OSError as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None:
            if operation_error is not None:
                raise CLIIOError(f"{operation_error}; input descriptor close failed: {path}: {close_error}") from operation_error
            raise CLIIOError(f"cannot close input descriptor for {path}: {close_error}") from close_error
        if operation_error is not None:
            raise operation_error
    return strict_json_loads(raw)


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
        except (OSError, NotImplementedError) as exc:
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
    close_errors: list[OSError] = []
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
                except OSError as exc:
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
        if staged_descriptor is not None:
            try:
                os.close(staged_descriptor)
            except OSError as exc:
                close_errors.append(exc)
        try:
            os.close(directory_descriptor)
        except OSError as exc:
            close_errors.append(exc)

    if close_errors:
        close_detail = "; ".join(str(error) for error in close_errors)
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
    value = _read_json(input_path)
    workload_value = _read_json(args.workload) if args.workload is not None else None
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


def _exit_for_error(error: AutoMLXError) -> int:
    return EXIT_CONTRACT


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        if args.command in _DEFERRED_COMMANDS:
            _write_diagnostic(
                _diagnostic(
                    "unavailable",
                    f"{args.command} is deferred: {_DEFERRED_COMMANDS[args.command]}",
                    details={
                        "status": "deferred",
                        "stage": _DEFERRED_STAGES[args.command],
                        "surface": "cli_orchestration",
                    },
                )
            )
            return EXIT_UNAVAILABLE
        result = _run_document_command(args)
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
