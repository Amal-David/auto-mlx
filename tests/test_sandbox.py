from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import CandidateProposal, FrozenWorkload, Knob
from auto_mlx.canonical import sha256_hex
from auto_mlx.errors import AutoMLXError
from auto_mlx.executor import (
    CleanupMode,
    ExecutionPolicy,
    ExecutionStatus,
    IsolatedProcess,
    IsolationClaim,
    IsolationProvider,
    TrustedRunner,
    TrustedRunnerRegistry,
    build_execution_plan,
    local_sandbox_primitives_available,
)
from auto_mlx.sandbox import (
    LocalSandboxAuthority,
    LocalSandboxProvider,
    ProbeEvidence,
    run_local_sandbox_probes,
    sandbox_profile_text,
)

_PRIMITIVES_AVAILABLE = local_sandbox_primitives_available()
_SKIP_REASON = "local sandbox-exec primitives (macOS + sandbox-exec) are unavailable on this host"


class _PermissiveSandboxProvider(LocalSandboxProvider):
    """Test double: launches through sandbox-exec with a sabotaged, permissive profile.

    Still follows the real ``sandbox-exec -p <text> argv...`` launch shape
    (so the authority can recover the profile text from ``Popen.args`` and
    the claim digest still matches it honestly) -- but the profile itself
    grants everything.  Used to prove the authority's probes, not just the
    launch shape, gate acceptance.
    """

    def enforce(self, argv, *, cwd, env, stdin, stdout, stderr) -> IsolatedProcess:
        sandbox_exec = shutil.which("sandbox-exec")
        assert sandbox_exec is not None
        scoped_dir = str(Path(cwd).resolve(strict=True))
        escaped = scoped_dir.replace("\\", "\\\\").replace('"', '\\"')
        profile_text = (
            "(version 1)\n"
            "(allow default)\n"
            f'(allow file-write* (subpath "{escaped}"))\n'
        )
        sandbox_argv = (sandbox_exec, "-p", profile_text, *argv)
        process = subprocess.Popen(
            sandbox_argv,
            cwd=cwd,
            env=dict(env),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=True,
        )
        return IsolatedProcess(process, self._claim(sha256_hex({"profile": profile_text})))


class _NonSandboxProvider(IsolationProvider):
    """A well-formed but non-local-sandbox provider; the authority must reject it."""

    def __init__(self) -> None:
        super().__init__("other-provider", "1" * 64, supports_evaluator_owned_launch=True)

    def enforce(self, argv, **kwargs) -> IsolatedProcess:
        process = subprocess.Popen(
            argv,
            cwd=kwargs["cwd"],
            env=dict(kwargs["env"]),
            stdin=kwargs["stdin"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            shell=False,
            start_new_session=(os.name == "posix"),
        )
        return IsolatedProcess(process, self._claim("2" * 64))


@unittest.skipUnless(_PRIMITIVES_AVAILABLE, _SKIP_REASON)
class LocalSandboxRealExecutionTests(unittest.TestCase):
    """These tests exercise real subprocess execution under sandbox-exec."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workload = FrozenWorkload("sandbox-fixture", knobs=(Knob("mode", "enum", values=("safe",)),))
        self.proposal = CandidateProposal("fixture-provider", self.workload, {"mode": "safe"})
        self.provider = LocalSandboxProvider()
        self.authority = LocalSandboxAuthority()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _plan(self, source: str, runner_id: str = "sandbox-runner"):
        script = self.root / f"{runner_id}.py"
        script.write_text(source, encoding="utf-8")
        runner = TrustedRunner.from_command(
            runner_id,
            (sys.executable, str(script)),
            artifact_paths=(str(script), str(Path(sys.executable).resolve())),
        )
        registry = TrustedRunnerRegistry((runner,))
        plan = build_execution_plan(self.proposal, registry, runner_id, str(self.root))
        return plan, registry

    def _policy(self, **kwargs) -> ExecutionPolicy:
        return ExecutionPolicy(timeout_seconds=5, max_output_bytes=1_048_576, **kwargs)

    def test_real_sandboxed_execution_produces_output_and_verified_evidence(self) -> None:
        plan, registry = self._plan("print('sandboxed-ok')\n")
        record = plan.execute(
            self._policy(),
            registry=registry,
            provider=self.provider,
            authority=self.authority,
            observation_id="s1",
            arm="baseline",
        )
        self.assertIs(record.status, ExecutionStatus.SUCCESS)
        self.assertEqual(record.stdout, b"sandboxed-ok\n")
        self.assertEqual(record.returncode, 0)
        self.assertIsNotNone(record.isolation)
        self.assertEqual(record.isolation.provider_id, "local-sandbox-exec")
        self.assertEqual(record.isolation.verifier_id, "local-sandbox-authority")
        self.assertTrue({"network_denial", "descendant_containment"}.issubset(record.isolation.requirements))
        self.assertTrue(record.promotion_eligible)
        self.assertGreater(record.parent_elapsed_ns, 0)

    def test_network_connect_is_denied_inside_the_sandbox(self) -> None:
        source = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(1.0)\n"
            "try:\n"
            "    s.connect(('192.0.2.1', 80))\n"
            "    print('network=allowed')\n"
            "except OSError as exc:\n"
            "    print('network=denied:' + type(exc).__name__)\n"
        )
        plan, registry = self._plan(source, "network-runner")
        record = plan.execute(self._policy(), registry=registry, provider=self.provider, authority=self.authority)
        self.assertIs(record.status, ExecutionStatus.SUCCESS)
        self.assertTrue(record.stdout.startswith(b"network=denied:"), record.stdout)

    def test_write_outside_scoped_directory_is_denied(self) -> None:
        outside = self.root / "outside-target.txt"
        source = (
            "import sys\n"
            f"target = {str(outside)!r}\n"
            "try:\n"
            "    open(target, 'w').write('escaped')\n"
            "    print('write=allowed')\n"
            "except OSError as exc:\n"
            "    print('write=denied:' + type(exc).__name__)\n"
        )
        # The runner script and its containing directory (self.root) are
        # NOT the execute_plan-managed scoped working directory (a fresh
        # temp dir per execution), so a write to a path under self.root is
        # a write outside the sandbox's scope.
        plan, registry = self._plan(source, "escape-runner")
        record = plan.execute(self._policy(), registry=registry, provider=self.provider, authority=self.authority)
        self.assertIs(record.status, ExecutionStatus.SUCCESS)
        self.assertTrue(record.stdout.startswith(b"write=denied:"), record.stdout)
        self.assertFalse(outside.exists())

    def test_timeout_kills_the_sandboxed_process_group(self) -> None:
        # The grandchild's pid is reported over stdout (captured incrementally
        # regardless of the eventual timeout), not written to a file: a file
        # write would have to land inside the sandbox's scoped working
        # directory, whose path this runner does not know in advance.
        source = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print('child_pid=' + str(child.pid), flush=True)\n"
            "while True: time.sleep(0.02)\n"
        )
        plan, registry = self._plan(source, "timeout-runner")
        # timeout_seconds is generous (not tight): a sandboxed process
        # spawning ANOTHER sandboxed process pays real, measured Seatbelt
        # mediation overhead (a second full interpreter start under policy
        # evaluation) that can take close to a second in practice. The
        # point of this test is the kill-and-cleanup behavior, not shaving
        # the timeout as thin as possible.
        record = plan.execute(
            ExecutionPolicy(timeout_seconds=3, kill_grace_seconds=0.5, max_output_bytes=4096),
            registry=registry,
            provider=self.provider,
            authority=self.authority,
        )
        self.assertIs(record.status, ExecutionStatus.TIMEOUT)
        self.assertEqual(record.cleanup.mode, CleanupMode.BEST_EFFORT_PROCESS_GROUP)
        self.assertFalse(record.cleanup.verified)
        self.assertTrue(record.stdout.startswith(b"child_pid="), record.stdout)
        child_pid = int(record.stdout.strip().split(b"=")[1])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except OSError:
                break
            time.sleep(0.02)
        else:
            self.fail("timed-out sandboxed child survived process-group cleanup")

    def test_output_cap_is_enforced_under_real_execution(self) -> None:
        plan, registry = self._plan(
            "import sys\nsys.stdout.write('x' * 1_000_000)\nsys.stdout.flush()\n",
            "flood-runner",
        )
        record = plan.execute(
            ExecutionPolicy(timeout_seconds=5, max_stdout_bytes=64, max_stderr_bytes=64, max_output_bytes=64),
            registry=registry,
            provider=self.provider,
            authority=self.authority,
        )
        self.assertIs(record.status, ExecutionStatus.OUTPUT_FAILURE)
        self.assertTrue(record.output_truncated)
        self.assertLessEqual(len(record.stdout), 64)

    def test_exit_and_crash_are_distinct_under_real_execution(self) -> None:
        exited, registry = self._plan("raise SystemExit(3)\n", "exit-runner")
        exited_record = exited.execute(self._policy(), registry=registry, provider=self.provider, authority=self.authority)
        self.assertIs(exited_record.status, ExecutionStatus.EXIT_FAILURE)
        self.assertEqual(exited_record.returncode, 3)

        crashed, crash_registry = self._plan(
            "import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n", "crash-runner"
        )
        crashed_record = crashed.execute(self._policy(), registry=crash_registry, provider=self.provider, authority=self.authority)
        self.assertIs(crashed_record.status, ExecutionStatus.CRASH)

    def test_authority_refuses_when_probe_profile_is_sabotaged(self) -> None:
        plan, registry = self._plan("print('must-not-be-trusted')\n", "sabotage-runner")
        record = plan.execute(
            self._policy(),
            registry=registry,
            provider=_PermissiveSandboxProvider(),
            authority=self.authority,
        )
        self.assertIs(record.status, ExecutionStatus.SANDBOX_UNAVAILABLE)
        self.assertEqual(record.failure.code.value, "sandbox_unavailable")

    def test_authority_rejects_a_foreign_provider(self) -> None:
        plan, registry = self._plan("print('must-not-be-trusted')\n", "foreign-runner")
        record = plan.execute(
            self._policy(),
            registry=registry,
            provider=_NonSandboxProvider(),
            authority=self.authority,
        )
        self.assertIs(record.status, ExecutionStatus.SANDBOX_UNAVAILABLE)

    def test_authority_rejects_a_tampered_claim_digest(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            profile_text = sandbox_profile_text(str(Path(workdir).resolve()))
            argv = (shutil.which("sandbox-exec"), "-p", profile_text, sys.executable, "-c", "print('ok')")
            process = subprocess.Popen(
                argv,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                tampered_claim = IsolationClaim(
                    self.provider.provider_id,
                    self.provider.identity,
                    frozenset({"network_denial", "descendant_containment"}),
                    "0" * 64,
                )
                with self.assertRaises(AutoMLXError):
                    self.authority.verify(self.provider, process, tampered_claim)
            finally:
                process.kill()
                process.wait(timeout=2)
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()

    def test_probe_helper_reports_denied_denied_allowed_directly(self) -> None:
        with tempfile.TemporaryDirectory() as scoped:
            resolved = str(Path(scoped).resolve())
            profile_text = sandbox_profile_text(resolved)
            evidence = run_local_sandbox_probes(profile_text, resolved)
            self.assertIsInstance(evidence, ProbeEvidence)
            self.assertTrue(evidence.network_connect_denied, evidence.detail)
            self.assertTrue(evidence.write_outside_scope_denied, evidence.detail)
            self.assertTrue(evidence.write_inside_scope_allowed, evidence.detail)
            self.assertTrue(evidence.confirms_isolation)

    def test_profile_text_is_deterministic_and_hash_stable(self) -> None:
        first = sandbox_profile_text("/tmp/example-scope")
        second = sandbox_profile_text("/tmp/example-scope")
        self.assertEqual(first, second)
        self.assertEqual(sha256_hex({"profile": first}), sha256_hex({"profile": second}))
        different = sandbox_profile_text("/tmp/other-scope")
        self.assertNotEqual(first, different)


class LocalSandboxPrimitiveGateTests(unittest.TestCase):
    """These tests run on every host: they verify the fail-closed gate itself."""

    def test_primitives_available_is_false_without_sandbox_exec_on_path(self) -> None:
        with tempfile.TemporaryDirectory() as empty_path_dir:
            with patch.dict(os.environ, {"PATH": empty_path_dir}):
                self.assertFalse(local_sandbox_primitives_available())

    @unittest.skipUnless(sys.platform == "darwin", "darwin-only: PATH manipulation is the only lever on this platform")
    def test_provider_construction_fails_closed_without_sandbox_exec(self) -> None:
        with tempfile.TemporaryDirectory() as empty_path_dir:
            with patch.dict(os.environ, {"PATH": empty_path_dir}):
                with self.assertRaises(AutoMLXError) as caught:
                    LocalSandboxProvider()
                self.assertEqual(caught.exception.code.value, "sandbox_unavailable")

    @unittest.skipUnless(sys.platform == "darwin", "darwin-only: PATH manipulation is the only lever on this platform")
    def test_authority_construction_fails_closed_without_sandbox_exec(self) -> None:
        with tempfile.TemporaryDirectory() as empty_path_dir:
            with patch.dict(os.environ, {"PATH": empty_path_dir}):
                with self.assertRaises(AutoMLXError) as caught:
                    LocalSandboxAuthority()
                self.assertEqual(caught.exception.code.value, "sandbox_unavailable")

    def test_execute_plan_stays_sandbox_unavailable_without_primitives_regardless_of_provider(self) -> None:
        # execute_plan's host-primitive gate is checked before it even
        # inspects the provider/authority it was handed, so a host without
        # the primitives behaves identically (SANDBOX_UNAVAILABLE) no
        # matter which provider a caller constructed elsewhere.
        temp = tempfile.TemporaryDirectory()
        try:
            root = Path(temp.name)
            workload = FrozenWorkload("gate-fixture", knobs=(Knob("mode", "enum", values=("safe",)),))
            proposal = CandidateProposal("fixture-provider", workload, {"mode": "safe"})
            script = root / "runner.py"
            script.write_text("print('must-not-run')\n", encoding="utf-8")
            runner = TrustedRunner.from_command(
                "gate-runner",
                (sys.executable, str(script)),
                artifact_paths=(str(script), str(Path(sys.executable).resolve())),
            )
            registry = TrustedRunnerRegistry((runner,))
            plan = build_execution_plan(proposal, registry, "gate-runner", str(root))
            policy = ExecutionPolicy(timeout_seconds=1, max_output_bytes=4096)
            provider = LocalSandboxProvider() if _PRIMITIVES_AVAILABLE else None
            authority = LocalSandboxAuthority() if _PRIMITIVES_AVAILABLE else None
            with patch("auto_mlx.executor.local_sandbox_primitives_available", return_value=False):
                record = plan.execute(policy, registry=registry, provider=provider, authority=authority)
            self.assertIs(record.status, ExecutionStatus.SANDBOX_UNAVAILABLE)
            self.assertEqual(record.failure.code.value, "sandbox_unavailable")
            self.assertEqual(record.stdout, b"")
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
