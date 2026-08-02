from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob
from auto_mlx.evaluator import Evaluator, ObservationBundle
from auto_mlx.errors import ContractError, Failure, FailureCode
from auto_mlx.executor import ExecutionPolicy, ExecutionStatus, IsolationAuthority, IsolationClaim, IsolationProvider, IsolatedProcess, TrustedRunner, TrustedRunnerRegistry
from auto_mlx.oracle import ExactOutputOracle


class FixtureIsolationProvider(IsolationProvider):
    def __init__(self) -> None:
        super().__init__("evaluator-fixture-isolation", "6" * 64)

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
        return IsolatedProcess(process, self._claim("7" * 64))


class TestOnlyIsolationAuthority(IsolationAuthority):
    def __init__(self) -> None:
        super().__init__("evaluator-test-verifier", "8" * 64, production_eligible=False)

    def verify(self, provider, process, claim: IsolationClaim):
        return self._attest(provider, claim)


class EligibleFixtureAuthority(IsolationAuthority):
    """Synthetic positive-path evidence; this is not a production sandbox."""

    def __init__(self) -> None:
        super().__init__("evaluator-eligible-test-verifier", "a" * 64, production_eligible=True)

    def verify(self, provider, process, claim: IsolationClaim):
        return self._attest(provider, claim)


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workload = FrozenWorkload("evaluator-fixture", knobs=(Knob("mode", "enum", values=("safe",)),))
        self.proposal = CandidateProposal("fixture-provider", self.workload, {"mode": "safe"})
        self.script = self.root / "runner.py"
        self.script.write_text("print('ok')\n", encoding="utf-8")
        baseline = TrustedRunner.from_command(
            "baseline",
            (sys.executable, str(self.script)),
            artifact_paths=(str(self.script), str(Path(sys.executable).resolve())),
        )
        candidate = TrustedRunner.from_command(
            "candidate",
            (sys.executable, str(self.script)),
            artifact_paths=(str(self.script), str(Path(sys.executable).resolve())),
        )
        self.registry = TrustedRunnerRegistry((baseline, candidate))
        self.provider = FixtureIsolationProvider()
        self.authority = TestOnlyIsolationAuthority()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _evaluator(self, *, provider=None) -> Evaluator:
        return Evaluator(
            self.registry,
            baseline_runner_id="baseline",
            candidate_runner_id="candidate",
            oracle=ExactOutputOracle(b"ok\n"),
            artifact_root=str(self.root),
            policy=EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=1, max_output_bytes=4096),
            execution_policy=ExecutionPolicy(
                timeout_seconds=1,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                max_output_bytes=4096,
            ),
            provider=self.provider if provider is None else provider,
            authority=self.authority,
        )

    def test_evaluator_returns_only_complete_provenance_bound_observations(self) -> None:
        bundle = self._evaluator().evaluate(self.proposal)
        self.assertIsInstance(bundle, ObservationBundle)
        self.assertTrue(bundle.accepted)
        self.assertEqual(len(bundle.measurements.blocks), 2)
        self.assertEqual(len(bundle.measurements.raw_records), 8)
        self.assertEqual(len(bundle.warmups), 2)
        self.assertEqual(len(bundle.raw_records), 10)
        self.assertTrue(all(sample.oracle.matched for sample in bundle.measurements.raw_samples))
        self.assertEqual(bundle.isolation_provider_id, "evaluator-fixture-isolation")
        self.assertEqual(bundle.isolation_identity, "6" * 64)
        self.assertEqual(bundle.isolation_verifier_id, "evaluator-test-verifier")
        self.assertTrue(all(not record.isolation.production_eligible for record in bundle.raw_records if record.isolation))
        self.assertFalse(bundle.promotion_eligible)

    def test_failed_warmup_blocks_accepted_evidence(self) -> None:
        bundle = self._evaluator().evaluate(self.proposal)
        first = bundle.warmups[0]
        failed_record = replace(
            first.record,
            status=ExecutionStatus.EXIT_FAILURE,
            returncode=1,
            failure=Failure(FailureCode.RUNTIME_FAILURE, "warmup failed"),
        )
        failed = replace(
            bundle,
            warmups=(replace(first, record=failed_record),) + bundle.warmups[1:],
        )
        self.assertFalse(failed.accepted)
        self.assertFalse(failed.promotion_eligible)

    def test_block_count_and_execution_policy_must_match_declared_policy(self) -> None:
        with self.assertRaises(ContractError):
            Evaluator(
                self.registry,
                baseline_runner_id="baseline",
                candidate_runner_id="candidate",
                oracle=ExactOutputOracle(b"ok\n"),
                artifact_root=str(self.root),
                policy=EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=1, max_output_bytes=4096),
                execution_policy=ExecutionPolicy(timeout_seconds=1, max_output_bytes=4096),
                provider=self.provider,
                authority=self.authority,
                block_count=1,
            )
        with self.assertRaises(ContractError):
            Evaluator(
                self.registry,
                baseline_runner_id="baseline",
                candidate_runner_id="candidate",
                oracle=ExactOutputOracle(b"ok\n"),
                artifact_root=str(self.root),
                policy=EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=1, max_output_bytes=4096),
                execution_policy=ExecutionPolicy(
                    timeout_seconds=1,
                    max_stdout_bytes=64,
                    max_stderr_bytes=4096,
                    max_output_bytes=4096,
                ),
                provider=self.provider,
                authority=self.authority,
            )

    def test_missing_provider_fails_closed_and_records_unverified_samples(self) -> None:
        with self.assertRaises(ContractError):
            self._evaluator(provider=object())  # type: ignore[arg-type]

    def test_default_evaluation_without_provider_fails_closed(self) -> None:
        evaluator = Evaluator(
            self.registry,
            baseline_runner_id="baseline",
            candidate_runner_id="candidate",
            oracle=ExactOutputOracle(b"ok\n"),
            artifact_root=str(self.root),
            policy=EvaluationPolicy(warmup_runs=0, measurement_runs=1, timeout_seconds=1, max_output_bytes=4096),
            authority=None,
        )
        bundle = evaluator.evaluate(self.proposal)
        self.assertFalse(bundle.accepted)
        self.assertEqual(len(bundle.measurements.raw_records), 4)
        self.assertTrue(all(record.status is ExecutionStatus.SANDBOX_UNAVAILABLE for record in bundle.measurements.raw_records))
        self.assertTrue(any("isolation_unverified" in reason for reason in bundle.measurements.rejection_reasons))

    def test_complete_production_eligible_evidence_is_marked_promotion_eligible(self) -> None:
        evaluator = Evaluator(
            self.registry,
            baseline_runner_id="baseline",
            candidate_runner_id="candidate",
            oracle=ExactOutputOracle(b"ok\n"),
            artifact_root=str(self.root),
            policy=EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=2, max_output_bytes=4096),
            execution_policy=ExecutionPolicy(
                timeout_seconds=2,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                max_output_bytes=4096,
            ),
            provider=self.provider,
            authority=EligibleFixtureAuthority(),
        )
        bundle = evaluator.evaluate(self.proposal)
        self.assertTrue(bundle.accepted)
        self.assertTrue(bundle.promotion_eligible)


if __name__ == "__main__":
    unittest.main()
