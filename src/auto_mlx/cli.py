"""Dry-run-first, MLX-free command line interface for Auto MLX G0.

The CLI validates declarative documents and inspects identities.  Evaluator,
receipt, promotion, and dispatch libraries are available separately; their
CLI orchestration remains explicitly deferred until the required evidence and
activation boundaries are proven.
"""

from __future__ import annotations

import argparse
import os
import sys
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


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(message)


def _kind_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--output", help="create this canonical document file; an existing file is never overwritten")


def build_parser() -> argparse.ArgumentParser:
    parser = JSONArgumentParser(
        prog="auto-mlx",
        description="Validate canonical Auto MLX documents and inspect their identities without MLX.",
    )
    parser.add_argument("--version", action="version", version="auto-mlx 0.1.0")
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
        )
        _kind_arguments(command_parser)
    for command, reason in _DEFERRED_COMMANDS.items():
        subparsers.add_parser(command, help=f"deferred: {reason}")
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
    sys.stderr.write(canonical_json(value) + "\n")


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


def _read_json(path_value: str) -> Any:
    if path_value == "-":
        stream = getattr(sys.stdin, "buffer", sys.stdin)
        raw = _read_bounded(stream, source="stdin", text_stream=stream is sys.stdin)
    else:
        path = Path(path_value)
        if path.stat().st_size > MAX_JSON_INPUT_BYTES:
            raise _input_limit_error(str(path))
        with path.open("rb") as stream:
            raw = _read_bounded(stream, source=str(path))
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


def _create_only(path_value: str, payload: bytes) -> None:
    path = Path(path_value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ContractError(
            f"refusing to overwrite existing output: {path}",
            code=FailureCode.INVALID_VALUE,
        ) from exc
    except OSError as exc:
        raise AutoMLXError(f"cannot create output {path}: {exc}", code=FailureCode.INVALID_VALUE) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except OSError as exc:
        raise AutoMLXError(f"cannot write output {path}: {exc}", code=FailureCode.INVALID_VALUE) from exc


def _run_document_command(args: argparse.Namespace) -> dict[str, Any]:
    kind = _resolve_kind(args)
    input_path = _resolve_path(args)
    value = _read_json(input_path)
    workload_value = _read_json(args.workload) if args.workload is not None else None
    document = _as_document(kind, value, workload_value=workload_value, artifact_root=args.artifact_root)
    document_value = _to_dict(document)
    canonical = canonical_json(document_value)
    if args.output is not None:
        _create_only(args.output, canonical.encode("utf-8"))
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
    if args.output is not None:
        result["output"] = args.output
    return result


def _exit_for_error(error: AutoMLXError) -> int:
    return EXIT_CONTRACT


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
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
    except SystemExit as exc:
        # argparse uses SystemExit for normal --help/--version output. Keep
        # the public ``main`` helper return-oriented for embedding and tests.
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    except CLIUsageError as exc:
        _write_diagnostic(_diagnostic("usage_error", str(exc)))
        return EXIT_USAGE
    except AutoMLXError as exc:
        _write_diagnostic(_diagnostic(exc.code.value, exc.message))
        return _exit_for_error(exc)
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as exc:
        _write_diagnostic(_diagnostic("io_error", str(exc)))
        return EXIT_IO
    except Exception as exc:  # pragma: no cover - a final stable boundary for CLI callers
        _write_diagnostic(_diagnostic("internal_error", str(exc)))
        return EXIT_INTERNAL
    sys.stdout.write(canonical_json(result) + "\n")
    return EXIT_OK


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
