"""A real, honest, macOS-local sandbox tier for the G1 evaluator.

``LocalSandboxProvider`` launches the evaluator's trusted-runner argv under
``sandbox-exec`` (Seatbelt): outbound network denied, filesystem writes
scoped to the execution's own working directory, a fresh session/process
group, and conservative CPU/file-size/open-file resource limits.
``LocalSandboxAuthority`` never trusts the provider's self-reported claim --
it independently extracts the exact profile text the process was actually
launched with (from the documented, read-only ``Popen.args`` attribute) and
runs its own probe processes under that identical profile before minting
any evidence.

This is a **developer-grade, single-user local guard**, not a hardened
multi-tenant security boundary.  It is appropriate for evaluating your own
candidate configuration on your own Mac.  It is not appropriate for running
untrusted code from strangers, for isolating mutually-distrusting tenants,
or for any claim of production-grade sandboxing.  See
``docs/threat-model.md`` for the honest boundary this tier draws.

There is intentionally no MLX dependency here, and no CLI wiring: this
module only supplies an ``IsolationProvider``/``IsolationAuthority`` pair
that a caller must explicitly construct and pass to ``execute_plan`` (or an
``Evaluator``) to opt into real local execution.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .canonical import sha256_hex
from .errors import AutoMLXError, ContractError, FailureCode
from .executor import (
    IsolatedProcess,
    IsolationAuthority,
    IsolationClaim,
    IsolationProvider,
    VerifiedIsolation,
    local_sandbox_primitives_available,
)


_SANDBOX_EXEC_BINARY: Final = "sandbox-exec"
_DEFAULT_CPU_SECONDS: Final = 120
_DEFAULT_MAX_FILE_SIZE_BYTES: Final = 256 * 1024 * 1024
_DEFAULT_MAX_OPEN_FILES: Final = 256
_PROBE_TIMEOUT_SECONDS: Final = 2.0
# RFC 5737 TEST-NET-1: reserved for documentation, never routed. The probe
# never reaches the network either way (sandbox-exec denies the connect()
# syscall itself), but targeting a real address would be poor hygiene.
_PROBE_NETWORK_TARGET: Final = ("192.0.2.1", 80)

_NETWORK_PROBE_SCRIPT: Final = (
    "import socket, sys\n"
    "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
    "s.settimeout(1.0)\n"
    "try:\n"
    "    s.connect((sys.argv[1], int(sys.argv[2])))\n"
    "    print('ALLOWED')\n"
    "except OSError as exc:\n"
    "    print('DENIED:' + type(exc).__name__)\n"
    "finally:\n"
    "    s.close()\n"
)

_WRITE_PROBE_SCRIPT: Final = (
    "import sys\n"
    "try:\n"
    "    with open(sys.argv[1], 'w') as handle:\n"
    "        handle.write('probe')\n"
    "    print('ALLOWED')\n"
    "except OSError as exc:\n"
    "    print('DENIED:' + type(exc).__name__)\n"
)

_SUBPATH_PATTERN: Final = re.compile(r'\(allow file-write\* \(subpath "((?:[^"\\]|\\.)*)"\)\)')


def _escape_profile_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _unescape_profile_string(value: str) -> str:
    return value.replace('\\"', '"').replace("\\\\", "\\")


def sandbox_profile_text(scoped_write_dir: str) -> str:
    """Build a deterministic sandbox-exec (Seatbelt) profile.

    Denies everything by default, explicitly denies network, allows process
    fork/exec (children inherit these same restrictions -- macOS Seatbelt
    profiles cannot be escaped by forking), allows broad reads (interpreter
    and framework imports live all over the filesystem), and allows writes
    only under ``scoped_write_dir``.  Deterministic in its one input, so the
    profile text -- and its hash -- is reproducible and auditable.
    """

    if type(scoped_write_dir) is not str or not scoped_write_dir or not Path(scoped_write_dir).is_absolute():
        raise ContractError("scoped_write_dir must be an absolute path string", code=FailureCode.UNSAFE_PATH)
    escaped = _escape_profile_string(scoped_write_dir)
    return (
        "(version 1)\n"
        "(deny default)\n"
        "(deny network*)\n"
        "(allow process-fork)\n"
        "(allow process-exec)\n"
        "(allow file-read*)\n"
        f'(allow file-write* (subpath "{escaped}"))\n'
        "(allow sysctl-read)\n"
        "(allow mach-lookup)\n"
        "(allow iokit-open)\n"
    )


def _profile_digest(profile_text: str) -> str:
    return sha256_hex({"profile": profile_text})


def _require_sandbox_exec() -> str:
    if not local_sandbox_primitives_available():
        raise AutoMLXError(
            "local sandbox-exec primitives are unavailable on this host",
            code=FailureCode.SANDBOX_UNAVAILABLE,
        )
    sandbox_exec = shutil.which(_SANDBOX_EXEC_BINARY)
    if sandbox_exec is None:  # pragma: no cover - guarded by the check above
        raise AutoMLXError("sandbox-exec is unavailable", code=FailureCode.SANDBOX_UNAVAILABLE)
    return sandbox_exec


class LocalSandboxProvider(IsolationProvider):
    """Developer-grade macOS local sandbox via ``sandbox-exec`` (Seatbelt).

    NOT a hardened multi-tenant boundary -- see the module docstring.
    """

    def __init__(
        self,
        *,
        cpu_seconds: int = _DEFAULT_CPU_SECONDS,
        max_file_size_bytes: int = _DEFAULT_MAX_FILE_SIZE_BYTES,
        max_open_files: int = _DEFAULT_MAX_OPEN_FILES,
    ) -> None:
        for label, value in (
            ("cpu_seconds", cpu_seconds),
            ("max_file_size_bytes", max_file_size_bytes),
            ("max_open_files", max_open_files),
        ):
            if type(value) is not int or value <= 0:
                raise ContractError(f"{label} must be a positive integer", code=FailureCode.INVALID_POLICY)
        self._sandbox_exec = _require_sandbox_exec()
        self._cpu_seconds = cpu_seconds
        self._max_file_size_bytes = max_file_size_bytes
        self._max_open_files = max_open_files
        super().__init__(
            "local-sandbox-exec",
            sha256_hex({"provider": "local-sandbox-exec", "version": 1}),
            supports_evaluator_owned_launch=True,
        )

    def enforce(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        stdin: object,
        stdout: object,
        stderr: object,
    ) -> IsolatedProcess:
        # Re-checked here (not just at construction time): primitives are a
        # runtime host property that a caller could hold a provider instance
        # across, and this method must never launch anything unsandboxed.
        sandbox_exec = _require_sandbox_exec()
        try:
            scoped_dir = str(Path(cwd).resolve(strict=True))
        except OSError as exc:
            raise AutoMLXError(f"sandbox working directory could not be resolved: {exc}", code=FailureCode.SANDBOX_UNAVAILABLE) from exc
        profile_text = sandbox_profile_text(scoped_dir)
        sandbox_argv = (sandbox_exec, "-p", profile_text, *argv)
        # Computed BEFORE spawning: if claim construction ever raises (e.g.
        # a misbehaving provider_id/identity override), nothing has been
        # started yet, so execute_plan's caller never has to worry about a
        # process leaked by a partially-failed enforce() call.
        claim = self._claim(_profile_digest(profile_text))

        cpu_seconds = self._cpu_seconds
        max_file_size_bytes = self._max_file_size_bytes
        max_open_files = self._max_open_files

        def _apply_resource_limits() -> None:
            # Runs in the forked child, before exec.  Deliberately minimal:
            # only stdlib resource.setrlimit calls, which never touch the
            # GIL/import machinery a concurrently multi-threaded parent
            # could otherwise deadlock a preexec_fn on after fork().
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_size_bytes, max_file_size_bytes))
            resource.setrlimit(resource.RLIMIT_NOFILE, (max_open_files, max_open_files))

        process = subprocess.Popen(
            sandbox_argv,
            cwd=cwd,
            env=dict(env),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=True,
            preexec_fn=_apply_resource_limits,
        )
        return IsolatedProcess(process, claim)


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    """What the authority's own probes attempted and what the OS returned."""

    network_connect_denied: bool
    write_outside_scope_denied: bool
    write_inside_scope_allowed: bool
    detail: Mapping[str, str]

    @property
    def confirms_isolation(self) -> bool:
        return self.network_connect_denied and self.write_outside_scope_denied and self.write_inside_scope_allowed


def _extract_profile_text(process: subprocess.Popen) -> str:
    """Recover the literal profile text a process was launched with.

    ``Popen.args`` is a documented, read-only attribute holding the exact
    argv passed to the constructor.  Since ``LocalSandboxProvider`` always
    launches via ``sandbox-exec -p <profile-text> <argv...>``, this is not a
    provider self-report -- it is the OS-launch-relevant argv itself.
    """

    args = getattr(process, "args", None)
    if not isinstance(args, (list, tuple)) or len(args) < 3:
        raise AutoMLXError(
            "process was not launched with a recoverable sandbox-exec profile",
            code=FailureCode.SANDBOX_UNAVAILABLE,
        )
    exec_path, flag, profile_text = args[0], args[1], args[2]
    if Path(str(exec_path)).name != _SANDBOX_EXEC_BINARY:
        raise AutoMLXError("process was not launched through sandbox-exec", code=FailureCode.SANDBOX_UNAVAILABLE)
    if flag != "-p" or type(profile_text) is not str or not profile_text:
        raise AutoMLXError("process argv does not carry an inline sandbox-exec profile", code=FailureCode.SANDBOX_UNAVAILABLE)
    return profile_text


def _extract_scoped_subpath(profile_text: str) -> str:
    match = _SUBPATH_PATTERN.search(profile_text)
    if match is None:
        raise AutoMLXError("sandbox profile does not declare a scoped write subpath", code=FailureCode.SANDBOX_UNAVAILABLE)
    return _unescape_profile_string(match.group(1))


def _run_probe_script(sandbox_exec: str, profile_text: str, script: str, args: Sequence[str], *, cwd: str) -> str:
    argv = [sandbox_exec, "-p", profile_text, sys.executable, "-c", script, *args]
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AutoMLXError(
            f"local sandbox probe process failed: {type(exc).__name__}",
            code=FailureCode.SANDBOX_UNAVAILABLE,
        ) from exc
    output = completed.stdout.decode("utf-8", errors="replace").strip()
    if not output.startswith(("ALLOWED", "DENIED")):
        raise AutoMLXError(
            "local sandbox probe produced an unexpected result",
            code=FailureCode.SANDBOX_UNAVAILABLE,
        )
    return output


def run_local_sandbox_probes(profile_text: str, scoped_dir: str) -> ProbeEvidence:
    """Independently probe an active sandbox-exec profile.

    Runs three separate, bounded subprocess invocations under the given
    profile text: an outbound TCP connect attempt, a write outside
    ``scoped_dir``, and a write inside it.  Any probe anomaly (an
    unexpected allow, a crash, a timeout, a missing sandbox-exec binary)
    raises a typed ``AutoMLXError`` -- this function never returns a silent
    guess.
    """

    sandbox_exec = _require_sandbox_exec()
    probe_root = tempfile.mkdtemp(prefix=".auto-mlx-authority-probe-")
    inside_target = str(Path(scoped_dir) / f".auto_mlx_authority_probe_{os.getpid()}_{id(probe_root)}")
    try:
        network_result = _run_probe_script(
            sandbox_exec,
            profile_text,
            _NETWORK_PROBE_SCRIPT,
            [_PROBE_NETWORK_TARGET[0], str(_PROBE_NETWORK_TARGET[1])],
            cwd=probe_root,
        )
        outside_target = str(Path(probe_root) / ".auto_mlx_probe_outside")
        outside_result = _run_probe_script(sandbox_exec, profile_text, _WRITE_PROBE_SCRIPT, [outside_target], cwd=probe_root)
        inside_result = _run_probe_script(sandbox_exec, profile_text, _WRITE_PROBE_SCRIPT, [inside_target], cwd=probe_root)
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)
        try:
            os.remove(inside_target)
        except OSError:
            pass
    return ProbeEvidence(
        network_connect_denied=network_result.startswith("DENIED"),
        write_outside_scope_denied=outside_result.startswith("DENIED"),
        write_inside_scope_allowed=inside_result.startswith("ALLOWED"),
        detail={
            "network_connect": network_result,
            "write_outside_scope": outside_result,
            "write_inside_scope": inside_result,
        },
    )


class LocalSandboxAuthority(IsolationAuthority):
    """Independent verifier that never trusts a provider's self-reported claim.

    Before minting any evidence, this authority: (1) recovers the exact
    profile text the process was actually launched with (from ``Popen.args``,
    not a provider self-report), (2) confirms the provider's claim digest
    really is a hash of that same text, and (3) runs its own probe
    processes -- an outbound connect attempt, a write outside the scoped
    subpath, and a write inside it -- under that identical profile.  Only a
    positive result on all three probes produces ``VerifiedIsolation``; any
    anomaly is a typed, fail-closed error, never a claim.
    """

    def __init__(self) -> None:
        if not local_sandbox_primitives_available():
            raise AutoMLXError(
                "local sandbox-exec primitives are unavailable on this host",
                code=FailureCode.SANDBOX_UNAVAILABLE,
            )
        super().__init__(
            "local-sandbox-authority",
            sha256_hex({"authority": "local-sandbox-authority", "version": 1}),
            production_eligible=False,
        )

    def verify(
        self,
        provider: IsolationProvider,
        process: subprocess.Popen,
        claim: IsolationClaim,
    ) -> VerifiedIsolation:
        if not local_sandbox_primitives_available():
            raise AutoMLXError(
                "local sandbox-exec primitives became unavailable",
                code=FailureCode.SANDBOX_UNAVAILABLE,
            )
        if not isinstance(provider, LocalSandboxProvider):
            raise AutoMLXError(
                "local sandbox authority can only verify a LocalSandboxProvider",
                code=FailureCode.SANDBOX_UNAVAILABLE,
            )
        if claim.provider_id != provider.provider_id or claim.provider_identity != provider.identity:
            raise AutoMLXError("isolation claim provider identity mismatch", code=FailureCode.SANDBOX_UNAVAILABLE)

        profile_text = _extract_profile_text(process)
        if _profile_digest(profile_text) != claim.attestation_digest:
            raise AutoMLXError(
                "isolation claim does not match the profile actually used to launch the process",
                code=FailureCode.SANDBOX_UNAVAILABLE,
            )
        scoped_dir = _extract_scoped_subpath(profile_text)
        evidence = run_local_sandbox_probes(profile_text, scoped_dir)
        if not evidence.confirms_isolation:
            raise AutoMLXError(
                "local sandbox probe did not confirm isolation enforcement",
                code=FailureCode.SANDBOX_UNAVAILABLE,
            )
        evidence_digest = sha256_hex(
            {
                "tier": provider.provider_id,
                "profile_sha256": hashlib.sha256(profile_text.encode("utf-8")).hexdigest(),
                "probe_evidence": dict(evidence.detail),
            }
        )
        # Constructed directly (not via self._attest) because the evidence
        # digest here binds the probe outcome, not merely the provider's
        # self-reported claim digest; the provider-identity check above is
        # the same one _attest performs.
        return VerifiedIsolation(
            provider.provider_id,
            provider.identity,
            self.verifier_id,
            self.identity,
            claim.requirements,
            evidence_digest,
            self.production_eligible,
            _authority=self,
        )


__all__: Final = [
    "LocalSandboxAuthority",
    "LocalSandboxProvider",
    "ProbeEvidence",
    "run_local_sandbox_probes",
    "sandbox_profile_text",
]
