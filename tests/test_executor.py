from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import Artifact, CandidateProposal, FrozenWorkload, Knob
from auto_mlx.errors import AutoMLXError, ContractError
from auto_mlx.executor import (
    CleanupMode,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionStatus,
    IsolationAuthority,
    IsolationClaim,
    IsolationProvider,
    IsolatedProcess,
    TrustedRunner,
    TrustedRunnerRegistry,
    build_execution_plan,
)
import auto_mlx.executor as executor_module


class FixtureIsolationProvider(IsolationProvider):
    """Test double for a separately supplied real sandbox integration."""

    def __init__(self) -> None:
        super().__init__("fixture-isolation", "1" * 64)

    def enforce(self, argv, *, cwd, env, stdin, stdout, stderr) -> IsolatedProcess:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(env),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            start_new_session=(os.name == "posix"),
        )
        return IsolatedProcess(process, self._claim("8" * 64))


class TestOnlyIsolationAuthority(IsolationAuthority):
    """Explicit test-only verifier; its evidence is never production evidence."""

    def __init__(self) -> None:
        super().__init__("test-only-verifier", "9" * 64, production_eligible=False)

    def verify(self, provider, process, claim: IsolationClaim):
        return self._attest(provider, claim)


class UnavailableProvider(IsolationProvider):
    def __init__(self) -> None:
        super().__init__("unavailable-test", "2" * 64)

    def enforce(self, argv, **kwargs) -> IsolatedProcess:
        raise AutoMLXError("test provider cannot enforce containment", code=executor_module.FailureCode.SANDBOX_UNAVAILABLE)


class ExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workload = FrozenWorkload("executor-fixture", knobs=(Knob("mode", "enum", values=("safe", "other")),))
        self.proposal = CandidateProposal("fixture-provider", self.workload, {"mode": "safe"})
        self.provider = FixtureIsolationProvider()
        self.authority = TestOnlyIsolationAuthority()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _runner(self, source: str, runner_id: str = "fixture") -> tuple[TrustedRunner, TrustedRunnerRegistry]:
        script = self.root / f"{runner_id}.py"
        script.write_text(source, encoding="utf-8")
        runner = TrustedRunner.from_command(
            runner_id,
            (sys.executable, str(script)),
            artifact_paths=(str(script), str(Path(sys.executable).resolve())),
        )
        return runner, TrustedRunnerRegistry((runner,))

    def _plan(self, source: str, runner_id: str = "fixture") -> ExecutionPlan:
        runner, registry = self._runner(source, runner_id)
        self.registry = registry
        return build_execution_plan(self.proposal, registry, runner.runner_id, str(self.root))

    def _execute(self, plan: ExecutionPlan, policy: ExecutionPolicy, **kwargs):
        registry = kwargs.pop("registry", getattr(self, "registry", None))
        authority = kwargs.pop("authority", self.authority)
        return plan.execute(policy, registry=registry, authority=authority, **kwargs)

    def _policy(self, **kwargs) -> ExecutionPolicy:
        return ExecutionPolicy(timeout_seconds=1, max_output_bytes=4096, **kwargs)

    def test_plan_uses_registered_argv_and_typed_config_only(self) -> None:
        plan = self._plan("print('ok')\n")
        self.assertEqual(plan.argv[0], str(Path(sys.executable).resolve()))
        self.assertIn(b'"mode":"safe"', plan.config_bytes)

    def test_forged_bin_sh_plan_is_rejected_before_launch(self) -> None:
        runner, registry = self._runner("print('trusted')\n", "trusted")
        forged = ExecutionPlan(
            self.proposal.candidate_id,
            self.proposal.workload_hash,
            runner.runner_id,
            runner.digest,
            ("/bin/sh", "-c", "echo forged"),
            runner.artifacts,
            (),
            str(self.root),
            b"{}",
            "1" * 64,
        )
        record = forged.execute(
            self._policy(),
            registry=registry,
            provider=self.provider,
            authority=self.authority,
        )
        self.assertIs(record.status, ExecutionStatus.ARTIFACT_FAILURE)
        self.assertEqual(record.failure.code.value, "identity_mismatch")  # type: ignore[union-attr]

    def test_environment_is_allowlisted_and_working_directories_are_isolated(self) -> None:
        source = """import os
from pathlib import Path
print('secret=' + str('AUTO_MLX_TEST_SECRET' in os.environ))
print('home=' + str(Path(os.environ['HOME']).name == 'home'))
print('cache=' + str(Path(os.environ['XDG_CACHE_HOME']).name == 'cache'))
print('cwd=' + str(Path.cwd().name.startswith('.auto-mlx-')))
"""
        previous = os.environ.pop("AUTO_MLX_TEST_SECRET", None)
        try:
            record = self._execute(self._plan(source), self._policy(), provider=self.provider, observation_id="sample", arm="baseline")
        finally:
            if previous is not None:
                os.environ["AUTO_MLX_TEST_SECRET"] = previous
        self.assertIs(record.status, ExecutionStatus.SUCCESS)
        self.assertEqual(record.stdout, b"secret=False\nhome=True\ncache=True\ncwd=True\n")
        self.assertGreater(record.parent_elapsed_ns, 0)
        self.assertEqual(record.observation_id, "sample")
        self.assertIsNotNone(record.isolation)

    def test_frozen_artifacts_are_verified_and_staged_bytes_are_rechecked(self) -> None:
        frozen = self.root / "input.bin"
        frozen.write_bytes(b"frozen-bytes")
        artifact = Artifact.from_file(str(self.root), "input.bin")
        workload = FrozenWorkload("artifact-fixture", artifacts=(artifact,))
        proposal = CandidateProposal("fixture-provider", workload, {})
        script = self.root / "artifact-runner.py"
        script.write_text(
            "import hashlib, os, pathlib\nprint(hashlib.sha256((pathlib.Path(os.environ['AUTO_MLX_ARTIFACT_ROOT']) / 'input.bin').read_bytes()).hexdigest())\n",
            encoding="utf-8",
        )
        runner = TrustedRunner.from_command(
            "artifact-runner",
            (sys.executable, str(script)),
            artifact_paths=(str(script), str(Path(sys.executable).resolve())),
        )
        plan = build_execution_plan(proposal, TrustedRunnerRegistry((runner,)), runner.runner_id, str(self.root))
        self.registry = TrustedRunnerRegistry((runner,))
        record = self._execute(plan, self._policy(), provider=self.provider)
        self.assertIs(record.status, ExecutionStatus.SUCCESS)
        self.assertEqual(record.stdout, (artifact.sha256 + "\n").encode())

        original_copy = executor_module._copy_descriptor_bounded

        def tampering_copy(source_descriptor, destination, **kwargs):
            result = original_copy(source_descriptor, destination, **kwargs)
            if Path(destination).name == "input.bin":
                os.chmod(destination, 0o600)
                Path(destination).write_bytes(b"tampered")
                os.chmod(destination, 0o400)
            return result

        executor_module._copy_descriptor_bounded = tampering_copy
        try:
            tampered = self._execute(plan, self._policy(), provider=self.provider)
        finally:
            executor_module._copy_descriptor_bounded = original_copy
        self.assertIs(tampered.status, ExecutionStatus.ARTIFACT_FAILURE)
        self.assertEqual(tampered.failure.code.value, "artifact_size_mismatch")  # type: ignore[union-attr]

    def test_namespace_swap_during_copy_cannot_change_staged_evidence(self) -> None:
        source_file = self.root / "swap.bin"
        source_file.write_bytes(b"original")
        artifact = Artifact.from_file(str(self.root), "swap.bin")
        workload = FrozenWorkload("swap-fixture", artifacts=(artifact,))
        proposal = CandidateProposal("fixture-provider", workload, {})
        script = self.root / "swap-runner.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        runner = TrustedRunner.from_command(
            "swap-runner",
            (sys.executable, str(script)),
            artifact_paths=(str(script), str(Path(sys.executable).resolve())),
        )
        plan = build_execution_plan(proposal, TrustedRunnerRegistry((runner,)), runner.runner_id, str(self.root))
        self.registry = TrustedRunnerRegistry((runner,))
        original_open = executor_module._open_verified_file

        def namespace_swap(root, relative_path):
            descriptor = original_open(root, relative_path)
            if relative_path == "swap.bin":
                source_file.rename(source_file.with_name("parked.bin"))
                source_file.write_bytes(b"replacement")
            return descriptor

        executor_module._open_verified_file = namespace_swap
        try:
            record = self._execute(plan, self._policy(), provider=self.provider)
        finally:
            executor_module._open_verified_file = original_open
        self.assertIs(record.status, ExecutionStatus.SUCCESS)
        self.assertEqual(record.stdout, b"ok\n")

    def test_missing_absolute_runner_file_is_rejected_and_every_file_argument_is_bound(self) -> None:
        script = self.root / "registered.py"
        script.write_text("print('ok')\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            TrustedRunner.from_command(
                "missing",
                (sys.executable, str(self.root / "does-not-exist.py")),
                artifact_paths=(str(script), str(Path(sys.executable).resolve())),
            )
        with self.assertRaises(ContractError):
            TrustedRunner.from_command(
                "unbound",
                (sys.executable, str(script)),
                artifact_paths=(str(Path(sys.executable).resolve()),),
            )
        with self.assertRaises(ContractError):
            TrustedRunner.from_command(
                "relative-executable",
                ("python3", str(script)),
                artifact_paths=(str(script), str(Path(sys.executable).resolve())),
            )

    def test_execution_uses_pinned_runner_bytes_after_source_mutation(self) -> None:
        script = self.root / "pinned.py"
        script.write_text("print('original')\n", encoding="utf-8")
        runner = TrustedRunner.from_command(
            "pinned",
            (sys.executable, str(script)),
            artifact_paths=(str(script), str(Path(sys.executable).resolve())),
        )
        plan = build_execution_plan(self.proposal, TrustedRunnerRegistry((runner,)), runner.runner_id, str(self.root))
        self.registry = TrustedRunnerRegistry((runner,))

        class MutatingProvider(FixtureIsolationProvider):
            def enforce(inner_self, argv, *, cwd, env, stdin, stdout, stderr):
                script.write_text("print('mutable-source')\n", encoding="utf-8")
                return super().enforce(argv, cwd=cwd, env=env, stdin=stdin, stdout=stdout, stderr=stderr)

        record = self._execute(plan, self._policy(), provider=MutatingProvider())
        self.assertIs(record.status, ExecutionStatus.SUCCESS)
        self.assertEqual(record.stdout, b"original\n")

    def test_runner_digest_binds_actual_bytes_and_mutation_is_rejected(self) -> None:
        script = self.root / "mutable.py"
        script.write_text("print('first')\n", encoding="utf-8")
        runner = TrustedRunner.from_command(
            "mutable",
            (sys.executable, str(script)),
            artifact_paths=(str(script), str(Path(sys.executable).resolve())),
        )
        registry = TrustedRunnerRegistry((runner,))
        script.write_text("print('mutated')\n", encoding="utf-8")
        with self.assertRaises(ContractError):
            build_execution_plan(self.proposal, registry, "mutable", str(self.root))

    def test_timeout_cleanup_is_best_effort_and_does_not_certify_orphan_freedom(self) -> None:
        if os.name != "posix":
            self.skipTest("process-group assertion is POSIX-specific")
        pid_file = self.root / "child.pid"
        source = """import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
__import__('pathlib').Path(os.environ['AUTO_MLX_TEST_PID_PATH']).write_text(str(child.pid))
while True: time.sleep(.02)
"""
        record = self._execute(self._plan(source),
            ExecutionPolicy(timeout_seconds=1, max_output_bytes=4096, extra_environment={"AUTO_MLX_TEST_PID_PATH": str(pid_file)}),
            provider=self.provider,
        )
        self.assertIs(record.status, ExecutionStatus.TIMEOUT)
        self.assertEqual(record.cleanup.mode, CleanupMode.BEST_EFFORT_PROCESS_GROUP)
        self.assertFalse(record.cleanup.verified)
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except OSError:
                break
            time.sleep(0.02)
        else:
            self.fail("timed-out child process survived fixture group cleanup")

    def test_output_flood_is_bounded_and_explicitly_failed(self) -> None:
        record = self._execute(self._plan("import sys; sys.stdout.write('x' * 1000000); sys.stdout.flush()\n"),
            ExecutionPolicy(timeout_seconds=1, max_stdout_bytes=64, max_stderr_bytes=64, max_output_bytes=64),
            provider=self.provider,
        )
        self.assertIs(record.status, ExecutionStatus.OUTPUT_FAILURE)
        self.assertTrue(record.output_truncated)
        self.assertLessEqual(len(record.stdout), 64)

    def test_forged_capability_object_and_unavailable_provider_fail_closed(self) -> None:
        forged = self._plan("print('must-not-run')\n").execute(self._policy(), registry=self.registry, authority=self.authority, provider=object())  # type: ignore[arg-type]
        self.assertIs(forged.status, ExecutionStatus.SANDBOX_UNAVAILABLE)
        unavailable = self._execute(self._plan("print('must-not-run')\n"), self._policy(), provider=UnavailableProvider())
        self.assertIs(unavailable.status, ExecutionStatus.SANDBOX_UNAVAILABLE)

    def test_provider_self_claim_without_out_of_band_authority_fails_closed(self) -> None:
        record = self._plan("print('must-not-run')\n").execute(
            self._policy(),
            registry=self.registry,
            provider=self.provider,
            authority=None,
        )
        self.assertIs(record.status, ExecutionStatus.SANDBOX_UNAVAILABLE)
        self.assertIn("authority", record.failure.message)  # type: ignore[union-attr]

    def test_disabling_any_required_isolation_control_fails_closed_before_launch(self) -> None:
        record = self._execute(self._plan("print('must-not-run')\n"),
            ExecutionPolicy(
                timeout_seconds=1,
                max_output_bytes=4096,
                require_network_denial=True,
                require_descendant_containment=False,
            ),
            provider=self.provider,
        )
        self.assertIs(record.status, ExecutionStatus.SANDBOX_UNAVAILABLE)
        self.assertEqual(record.failure.code.value, "sandbox_unavailable")  # type: ignore[union-attr]
        self.assertEqual(record.stdout, b"")

    def test_timing_symlink_or_removal_is_ignored_and_parent_timing_is_only_timing(self) -> None:
        source = """import os, pathlib
path = pathlib.Path.cwd() / '.auto_mlx_child_timing_ns'
try: path.symlink_to('/dev/null')
except FileExistsError: pass
print('child-timing-env=' + str('AUTO_MLX_CHILD_TIMING_PATH' in os.environ))
"""
        record = self._execute(self._plan(source), self._policy(), provider=self.provider)
        self.assertIs(record.status, ExecutionStatus.SUCCESS)
        self.assertGreater(record.parent_elapsed_ns, 0)
        self.assertNotIn("child_elapsed_ns", record.to_dict())
        self.assertEqual(record.stdout, b"child-timing-env=False\n")

    def test_exit_and_crash_are_distinct_records(self) -> None:
        exited = self._execute(self._plan("raise SystemExit(3)\n"), self._policy(), provider=self.provider)
        self.assertIs(exited.status, ExecutionStatus.EXIT_FAILURE)
        crashed = self._execute(self._plan("import os, signal; os.kill(os.getpid(), signal.SIGTERM)\n", "crasher"), self._policy(), provider=self.provider)
        self.assertIs(crashed.status, ExecutionStatus.CRASH)


if __name__ == "__main__":
    unittest.main()
