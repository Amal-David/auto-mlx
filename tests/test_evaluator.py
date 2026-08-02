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
        self.assertFalse(bundle.accepted)
        self.assertEqual(len(bundle.measurements.blocks), 2)
        self.assertEqual(len(bundle.measurements.raw_records), 8)
        self.assertEqual(len(bundle.warmups), 2)
        self.assertEqual(len(bundle.raw_records), 10)
        self.assertTrue(all(record.status is ExecutionStatus.SANDBOX_UNAVAILABLE for record in bundle.raw_records))
        self.assertEqual(bundle.oracle_descriptor.expected_digest, ExactOutputOracle(b"ok\n").expected_digest)
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

    def test_acceptance_recomputes_measurements_and_oracle_binding(self) -> None:
        bundle = self._evaluator().evaluate(self.proposal)
        forged_summary = replace(
            bundle,
            measurements=replace(bundle.measurements, accepted=True, rejection_reasons=()),
        )
        self.assertFalse(forged_summary.accepted)
        forged_oracle = replace(
            bundle,
            oracle_descriptor=ExactOutputOracle(b"attacker\n").descriptor,
        )
        self.assertFalse(forged_oracle.accepted)

    def test_authority_eligibility_flag_is_ignored_and_oracle_is_retained(self) -> None:
        authority = EligibleFixtureAuthority()
        self.assertFalse(authority.production_eligible)
        evaluator = self._evaluator()
        bundle = evaluator.evaluate(self.proposal)
        self.assertIs(bundle.oracle, evaluator._oracle)
        self.assertEqual(bundle.to_dict()["oracle_descriptor"], bundle.oracle_descriptor.to_dict())

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

    def test_caller_created_eligible_authority_cannot_promote_g0_evidence(self) -> None:
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
        self.assertFalse(bundle.accepted)
        self.assertFalse(bundle.promotion_eligible)

    def test_promotion_eligibility_recomputes_reconstructed_evidence_bindings(self) -> None:
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
        self.assertFalse(bundle.promotion_eligible)

        first_warmup = bundle.warmups[0]
        bad_warmup_identity = replace(
            bundle,
            warmups=(replace(first_warmup, sample_id="forged-warmup"),) + bundle.warmups[1:],
        )
        bad_warmup_record = replace(
            first_warmup.record,
            observation_id="forged-warmup",
        )
        bad_warmup_provenance = replace(
            bundle,
            warmups=(replace(first_warmup, record=bad_warmup_record),) + bundle.warmups[1:],
        )
        bad_warmup_oracle = replace(
            bundle,
            warmups=(replace(first_warmup, oracle=replace(first_warmup.oracle, expected_digest="0" * 64)),) + bundle.warmups[1:],
        )
        bad_measurements = replace(bundle, measurements=replace(bundle.measurements, plan_digest="0" * 64))
        bad_count = replace(bundle, measurement_block_count=bundle.measurement_block_count + 1)  # type: ignore[operator]
        bad_policy = replace(bundle, evaluation_policy=None)
        bad_candidate = replace(bundle, candidate_id="f" * 64)
        bad_workload = replace(bundle, workload_hash="e" * 64)
        bad_runner = replace(bundle, baseline_runner_digest="d" * 64)
        bad_execution_digest = replace(bundle, execution_policy_digest="c" * 64)

        for name, forged in (
            ("warmup identity", bad_warmup_identity),
            ("warmup provenance", bad_warmup_provenance),
            ("warmup oracle", bad_warmup_oracle),
            ("measurement plan digest", bad_measurements),
            ("measurement count", bad_count),
            ("missing evaluation policy", bad_policy),
            ("candidate identity", bad_candidate),
            ("workload identity", bad_workload),
            ("runner identity", bad_runner),
            ("execution policy digest", bad_execution_digest),
        ):
            with self.subTest(name=name):
                self.assertFalse(forged.promotion_eligible)


if __name__ == "__main__":
    unittest.main()
