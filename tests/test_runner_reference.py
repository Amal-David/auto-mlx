"""Real, end-to-end tests for the reference MLX matmul runner.

Two families of test live here:

- ``ReferenceMatmulImportHygieneTests`` runs on every host, MLX or not: it
  proves (via clean subprocesses, not a source grep) that importing
  ``auto_mlx``, importing ``auto_mlx.runners.reference_matmul`` on its own,
  and running the CLI's ``validate`` command never put an ``mlx`` module
  into ``sys.modules``.
- Everything else requires MLX importable AND the macOS local
  ``sandbox-exec`` primitives (``local_sandbox_primitives_available()``);
  those tests run real subprocess executions of the reference runner
  under the real ``LocalSandboxProvider``/``LocalSandboxAuthority`` tier --
  no mocks -- and are skipped everywhere else.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob
from auto_mlx.evaluator import Evaluator
from auto_mlx.executor import (
    ExecutionPolicy,
    ExecutionStatus,
    TrustedRunnerRegistry,
    build_execution_plan,
    local_sandbox_primitives_available,
)
from auto_mlx.oracle import ExactOutputOracle
from auto_mlx.receipts import Receipt
from auto_mlx.runners import (
    BASELINE_RUNNER_ID,
    CANDIDATE_RUNNER_ID,
    reference_matmul_script_path,
    register_reference_matmul_runners,
)
from auto_mlx.runners.reference_matmul import WARMUP_MARKER
from auto_mlx.sandbox import LocalSandboxAuthority, LocalSandboxProvider


try:
    import mlx.core as _mlx_probe  # noqa: F401  -- availability probe only

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

_PRIMITIVES_AVAILABLE = local_sandbox_primitives_available()
_RUN_REAL = _MLX_AVAILABLE and _PRIMITIVES_AVAILABLE
_SKIP_REASON = "requires MLX installed and macOS local sandbox-exec primitives"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _toy_matmul_workload(name: str) -> FrozenWorkload:
    return FrozenWorkload(
        name,
        knobs=(
            Knob("mode", "enum", values=("eager", "compiled")),
            Knob("tile", "integer", minimum=16, maximum=32),
        ),
        parameters={"dtype": "float32", "shape": [1, 3072, 3072]},
    )


class ReferenceMatmulImportHygieneTests(unittest.TestCase):
    """These run everywhere: importing our package must never import mlx."""

    def _probe(self, snippet: str) -> list[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        return completed.stdout.splitlines()

    def test_importing_auto_mlx_does_not_import_mlx(self) -> None:
        lines = self._probe("import sys\nimport auto_mlx\nprint('mlx' in sys.modules)\n")
        self.assertEqual(lines, ["False"])

    def test_importing_reference_matmul_module_alone_does_not_import_mlx(self) -> None:
        lines = self._probe(
            "import sys\nimport auto_mlx.runners.reference_matmul\nprint('mlx' in sys.modules)\n"
        )
        self.assertEqual(lines, ["False"])

    def test_importing_runners_wiring_package_does_not_import_mlx(self) -> None:
        lines = self._probe("import sys\nimport auto_mlx.runners\nprint('mlx' in sys.modules)\n")
        self.assertEqual(lines, ["False"])

    def test_cli_validate_workload_does_not_import_mlx(self) -> None:
        lines = self._probe(
            "import sys\n"
            "from auto_mlx.cli import main\n"
            "status = main(['validate', 'workload', '--input', 'examples/workload.json'])\n"
            "print(status)\n"
            "print('mlx' in sys.modules)\n"
        )
        # main() itself writes the validation result JSON to stdout first;
        # the two lines this test cares about are the ones it appended.
        self.assertEqual(lines[-2:], ["0", "False"])
        payload = json.loads(lines[0])
        self.assertTrue(payload["ok"])


@unittest.skipUnless(_RUN_REAL, _SKIP_REASON)
class ReferenceMatmulRunnerDeterminismTests(unittest.TestCase):
    """Real sandboxed subprocess executions of reference_matmul.py."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = TrustedRunnerRegistry()
        register_reference_matmul_runners(self.registry)
        self.workload = _toy_matmul_workload("toy-matmul-determinism")
        self.provider = LocalSandboxProvider()
        self.authority = LocalSandboxAuthority()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _execute(self, runner_id: str, proposal: CandidateProposal) -> bytes:
        plan = build_execution_plan(proposal, self.registry, runner_id, str(self.root))
        record = plan.execute(
            ExecutionPolicy(timeout_seconds=60, max_output_bytes=4096),
            registry=self.registry,
            provider=self.provider,
            authority=self.authority,
        )
        self.assertIs(record.status, ExecutionStatus.SUCCESS, record.failure)
        self.assertTrue(record.stdout, "runner produced no stdout")
        return record.stdout

    def test_baseline_runner_is_deterministic_across_ten_sandboxed_runs(self) -> None:
        # tile is validated but inert (see reference_matmul's docstring), and
        # mode is forced to eager by the baseline runner's pinned argv, so a
        # candidate that declares mode=compiled must still produce the same
        # baseline output.
        proposal = CandidateProposal("test-provider", self.workload, {"mode": "compiled", "tile": 16})
        digests = {self._execute(BASELINE_RUNNER_ID, proposal) for _ in range(10)}
        self.assertEqual(len(digests), 1, digests)

    def test_candidate_compiled_runner_is_deterministic_across_ten_sandboxed_runs(self) -> None:
        proposal = CandidateProposal("test-provider", self.workload, {"mode": "compiled", "tile": 32})
        digests = {self._execute(CANDIDATE_RUNNER_ID, proposal) for _ in range(10)}
        self.assertEqual(len(digests), 1, digests)

    def test_eager_and_compiled_produce_byte_identical_stdout(self) -> None:
        eager_proposal = CandidateProposal("test-provider", self.workload, {"mode": "eager", "tile": 16})
        compiled_proposal = CandidateProposal("test-provider", self.workload, {"mode": "compiled", "tile": 32})
        eager_out = self._execute(CANDIDATE_RUNNER_ID, eager_proposal)
        compiled_out = self._execute(CANDIDATE_RUNNER_ID, compiled_proposal)
        self.assertEqual(eager_out, compiled_out)

    def test_baseline_runner_forces_eager_regardless_of_configured_mode(self) -> None:
        compiled_config_proposal = CandidateProposal("test-provider", self.workload, {"mode": "compiled", "tile": 16})
        forced_baseline_out = self._execute(BASELINE_RUNNER_ID, compiled_config_proposal)
        explicit_eager_proposal = CandidateProposal("test-provider", self.workload, {"mode": "eager", "tile": 16})
        candidate_eager_out = self._execute(CANDIDATE_RUNNER_ID, explicit_eager_proposal)
        self.assertEqual(forced_baseline_out, candidate_eager_out)

    def test_tile_value_does_not_change_the_result_digest(self) -> None:
        low_tile = CandidateProposal("test-provider", self.workload, {"mode": "eager", "tile": 16})
        high_tile = CandidateProposal("test-provider", self.workload, {"mode": "eager", "tile": 32})
        self.assertEqual(
            self._execute(CANDIDATE_RUNNER_ID, low_tile),
            self._execute(CANDIDATE_RUNNER_ID, high_tile),
        )

    def _execute_record(self, runner_id: str, proposal: CandidateProposal):
        plan = build_execution_plan(proposal, self.registry, runner_id, str(self.root))
        record = plan.execute(
            ExecutionPolicy(timeout_seconds=60, max_output_bytes=4096),
            registry=self.registry,
            provider=self.provider,
            authority=self.authority,
        )
        self.assertIs(record.status, ExecutionStatus.SUCCESS, record.failure)
        return record

    def test_runner_emits_warmup_marker_on_stderr_and_leaves_stdout_oracle_sacred(self) -> None:
        """Runner contract (see reference_matmul's module docstring): one
        uncounted warmup runs before the measured run; its completion is
        announced only on stderr, and stdout stays exactly the one
        oracle-sacred digest line the warmup never touches.
        """

        proposal = CandidateProposal("test-provider", self.workload, {"mode": "eager", "tile": 16})
        record = self._execute_record(CANDIDATE_RUNNER_ID, proposal)
        marker = WARMUP_MARKER.encode("utf-8")
        self.assertEqual(record.stderr.count(marker), 1, record.stderr)
        stdout_lines = record.stdout.splitlines()
        self.assertEqual(len(stdout_lines), 1, record.stdout)
        self.assertNotIn(marker, record.stdout)

    def test_runner_elapsed_ns_excludes_authority_verification_probe_time(self) -> None:
        """The evidentiary runner span must be strictly shorter than the
        full-sample span on a real run, since the full span additionally
        pays the real ``LocalSandboxAuthority``'s three probe subprocesses
        plus artifact staging (see ``execute_plan`` in ``auto_mlx.executor``).
        """

        proposal = CandidateProposal("test-provider", self.workload, {"mode": "eager", "tile": 16})
        record = self._execute_record(CANDIDATE_RUNNER_ID, proposal)
        self.assertIsNotNone(record.runner_elapsed_ns)
        self.assertGreater(record.runner_elapsed_ns, 0)
        self.assertLess(record.runner_elapsed_ns, record.parent_elapsed_ns)


@unittest.skipUnless(_RUN_REAL, _SKIP_REASON)
class ReferenceMatmulMalformedConfigTests(unittest.TestCase):
    """The runner's OWN validation, exercised directly under the sandbox.

    ``CandidateProposal``/``Knob`` already reject an out-of-contract config
    before it can ever reach a runner through ``execute_plan`` -- so these
    write a raw, deliberately malformed config file and invoke the runner
    the same way ``LocalSandboxProvider`` would, to prove the runner's own
    fail-closed parsing is real and not just inherited from the contract
    layer above it.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.provider = LocalSandboxProvider()
        self.interpreter = str(Path(sys.executable).resolve())
        self.script = str(reference_matmul_script_path())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run_with_raw_config(self, raw_config: bytes) -> subprocess.CompletedProcess[bytes]:
        work = Path(tempfile.mkdtemp(dir=str(self.root)))
        config_path = work / "config.json"
        config_path.write_bytes(raw_config)
        env = {"PATH": os.defpath, "AUTO_MLX_CONFIG_PATH": str(config_path)}
        launched = self.provider.enforce(
            (self.interpreter, self.script),
            cwd=str(work),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process = launched.process
        stdout, stderr = process.communicate(timeout=30)
        return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)

    def test_unknown_knob_is_rejected(self) -> None:
        result = self._run_with_raw_config(b'{"mode": "eager", "tile": 16, "extra": 1}')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"unknown knob", result.stderr)

    def test_missing_knob_is_rejected(self) -> None:
        result = self._run_with_raw_config(b'{"mode": "eager"}')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"missing knob", result.stderr)

    def test_bad_mode_enum_is_rejected(self) -> None:
        result = self._run_with_raw_config(b'{"mode": "not-a-real-mode", "tile": 16}')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"mode must be one of", result.stderr)

    def test_out_of_range_tile_is_rejected(self) -> None:
        result = self._run_with_raw_config(b'{"mode": "eager", "tile": 999}')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"tile must be within", result.stderr)

    def test_wrong_type_tile_is_rejected(self) -> None:
        result = self._run_with_raw_config(b'{"mode": "eager", "tile": "16"}')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"tile must be an integer", result.stderr)

    def test_invalid_json_is_rejected(self) -> None:
        result = self._run_with_raw_config(b"not json at all")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"not valid JSON", result.stderr)


@unittest.skipUnless(_RUN_REAL, _SKIP_REASON)
class ReferenceMatmulEvaluatorLoopTests(unittest.TestCase):
    """Full library-level loop: Evaluator -> real sandbox -> receipt."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = TrustedRunnerRegistry()
        register_reference_matmul_runners(self.registry)
        self.workload = _toy_matmul_workload("toy-matmul-evaluator-loop")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_baseline_vs_compiled_candidate_is_accepted_and_promotion_eligible(self) -> None:
        proposal = CandidateProposal("test-provider", self.workload, {"mode": "compiled", "tile": 24})

        # The oracle's expected bytes come from one real baseline execution,
        # not a hardcoded literal, so this stays correct if the runner's
        # hardcoded matrix spec ever changes.
        probe_plan = build_execution_plan(proposal, self.registry, BASELINE_RUNNER_ID, str(self.root))
        probe_record = probe_plan.execute(
            ExecutionPolicy(timeout_seconds=60, max_output_bytes=4096),
            registry=self.registry,
            provider=LocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        self.assertIs(probe_record.status, ExecutionStatus.SUCCESS, probe_record.failure)
        oracle = ExactOutputOracle(probe_record.stdout)

        policy = EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=60, max_output_bytes=4096)
        evaluator = Evaluator(
            self.registry,
            baseline_runner_id=BASELINE_RUNNER_ID,
            candidate_runner_id=CANDIDATE_RUNNER_ID,
            oracle=oracle,
            artifact_root=str(self.root),
            policy=policy,
            provider=LocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        bundle = evaluator.evaluate(proposal)

        self.assertTrue(all(record.status is ExecutionStatus.SUCCESS for record in bundle.raw_records))
        self.assertTrue(bundle.accepted, bundle.measurements.rejection_reasons)
        self.assertTrue(bundle.promotion_eligible)
        self.assertTrue(bundle.measurements.accepted)
        self.assertTrue(bundle.measurements.promotion_eligible)
        self.assertEqual(bundle.isolation_provider_id, "local-sandbox-exec")
        self.assertEqual(bundle.isolation_verifier_id, "local-sandbox-authority")

        receipt = Receipt.from_observation_bundle(
            bundle,
            self.workload,
            proposal,
            policy,
            oracle=oracle,
            created_at_ns=1,
        )
        self.assertTrue(receipt.raw_samples)
        for sample in receipt.raw_samples:
            # Real wall-times only: positive and plausible (well under the
            # policy's 60s timeout expressed in nanoseconds). No speedup
            # claim is asserted -- this is a correctness/plumbing check,
            # not a performance benchmark.
            self.assertGreater(sample.duration_ns, 0)
            self.assertLess(sample.duration_ns, 60_000_000_000)

        # Wave A receipt fields, populated from a real evaluate() call (not
        # synthetic wire data): thermal annotations, one per measurement
        # block, and a warm_state note derived from real runner stderr.
        self.assertIsNotNone(receipt.evaluator_bundle)
        thermal_blocks = receipt.evaluator_bundle["thermal_blocks"]
        self.assertEqual(len(thermal_blocks), policy.measurement_runs)
        for entry in thermal_blocks:
            self.assertIn(entry["preflight"]["initial"]["state"], {"nominal", "throttled", "unknown"})
            self.assertIn(entry["preflight"]["final"]["state"], {"nominal", "throttled", "unknown"})
            self.assertEqual(entry["policy"], "tag")

        measurement_samples = [
            sample
            for block in receipt.evaluator_bundle["measurements"]["blocks"]
            for sample in block["samples"]
        ]
        self.assertTrue(measurement_samples)
        # Every real successful sample carries a runner_elapsed_ns strictly
        # less than parent_elapsed_ns -- the real authority.verify() probe
        # cost (three sandbox-exec subprocesses) landed outside the
        # evidentiary span, not inside it.
        runner_spans = []
        for sample in measurement_samples:
            record = sample["record"]
            self.assertIsNotNone(record)
            self.assertEqual(record["status"], "success")
            self.assertIsNotNone(record["runner_elapsed_ns"])
            self.assertGreater(record["runner_elapsed_ns"], 0)
            self.assertLess(record["runner_elapsed_ns"], record["parent_elapsed_ns"])
            runner_spans.append(record["runner_elapsed_ns"])
        # The in-runner warmup ran on at least one real sample this call --
        # the marker was emitted on real stderr and independently detected.
        self.assertTrue(
            any(sample["warm_state"]["in_runner_warmup_marker_present"] for sample in measurement_samples)
        )
        self.assertTrue(
            any(warmup["warm_state"]["in_runner_warmup_marker_present"] for warmup in receipt.evaluator_bundle["warmups"])
        )
        # Gain math reads the runner span, not the full-sample span: the
        # receipt's reported sums must equal the sum of the real
        # runner_elapsed_ns values recomputed here independently.
        gain = receipt.metrics["gain"]
        self.assertEqual(gain["baseline_sum_ns"] + gain["candidate_sum_ns"], sum(runner_spans))


if __name__ == "__main__":
    unittest.main()
