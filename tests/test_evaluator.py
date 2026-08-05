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
from auto_mlx.executor import (
    ExecutionPolicy,
    ExecutionStatus,
    IsolationAuthority,
    IsolationClaim,
    IsolationProvider,
    IsolatedProcess,
    TrustedRunner,
    TrustedRunnerRegistry,
    local_sandbox_primitives_available,
)
from auto_mlx.oracle import ExactOutputOracle
from auto_mlx.sandbox import LocalSandboxAuthority, LocalSandboxProvider
from auto_mlx.thermal import ThermalReading
from auto_mlx.thermal import thermal_preflight as _real_thermal_preflight


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


class ExplodingFixtureProvider(FixtureIsolationProvider):
    @property
    def provider_id(self):
        raise AssertionError("evaluator must not read provider_id")

    @property
    def identity(self):
        raise AssertionError("evaluator must not read provider identity")


class ExplodingFixtureAuthority(TestOnlyIsolationAuthority):
    @property
    def verifier_id(self):
        raise AssertionError("evaluator must not read verifier_id")

    @property
    def identity(self):
        raise AssertionError("evaluator must not read verifier identity")

    @property
    def production_eligible(self):
        raise AssertionError("evaluator must not read production eligibility")


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

    def _evaluator(self, *, provider=None, authority=None, policy=None, thermal_preflight=None) -> Evaluator:
        return Evaluator(
            self.registry,
            baseline_runner_id="baseline",
            candidate_runner_id="candidate",
            oracle=ExactOutputOracle(b"ok\n"),
            artifact_root=str(self.root),
            policy=policy
            or EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=1, max_output_bytes=4096),
            thermal_preflight=thermal_preflight,
            execution_policy=ExecutionPolicy(
                timeout_seconds=1,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                max_output_bytes=4096,
            ),
            provider=self.provider if provider is None else provider,
            authority=self.authority if authority is None else authority,
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
        self.assertIsNone(bundle.isolation_provider_id)
        self.assertIsNone(bundle.isolation_identity)
        self.assertIsNone(bundle.isolation_verifier_id)
        self.assertIsNone(bundle.isolation_verifier_identity)
        self.assertIsNone(bundle.isolation_requirements)
        self.assertIn("production_isolation_unavailable", bundle.measurements.rejection_reasons)
        self.assertFalse(bundle.promotion_eligible)

    def _counting_preflight(self, *, thermally_suspect: bool):
        """A fast, deterministic stand-in for the real ``pmset``-backed preflight.

        Reuses ``auto_mlx.thermal.thermal_preflight`` itself (not a hand-rolled
        dict) so the shape returned here is byte-identical to what a real
        preflight produces; only the prober and the retry sleep are faked so
        the test never touches real hardware state or waits 30s.
        """

        calls: list[None] = []
        reading = ThermalReading("throttled" if thermally_suspect else "nominal", 50 if thermally_suspect else 100, None, None, "")

        def preflight():
            calls.append(None)
            return _real_thermal_preflight(prober=lambda: reading, sleep=lambda seconds: None, retry_pause_seconds=30.0)

        return preflight, calls

    def test_thermal_preflight_runs_once_per_block_and_is_recorded(self) -> None:
        preflight, calls = self._counting_preflight(thermally_suspect=False)
        bundle = self._evaluator(thermal_preflight=preflight).evaluate(self.proposal)
        # Two measurement blocks (measurement_runs=2 in _evaluator's default
        # policy) -- the preflight must run once per block, not once for the
        # whole evaluate() call, since thermal state can drift block to block.
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(bundle.thermal_blocks), 2)
        for index, entry in enumerate(bundle.thermal_blocks):
            with self.subTest(block_index=index):
                self.assertEqual(entry["block_index"], index)
                self.assertEqual(entry["block_id"], f"block-{index + 1:04d}")
                self.assertEqual(entry["policy"], "tag")
                self.assertFalse(entry["refused"])
                self.assertFalse(entry["preflight"]["thermally_suspect"])
        # Nominal thermal state changes nothing about sample collection.
        self.assertEqual(len(bundle.measurements.raw_records), 8)

    def test_thermal_gate_policy_tag_proceeds_with_suspect_block_tagged(self) -> None:
        """Default policy ("tag"): a persistently throttled block still runs its samples."""

        preflight, calls = self._counting_preflight(thermally_suspect=True)
        bundle = self._evaluator(thermal_preflight=preflight).evaluate(self.proposal)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(not entry["refused"] for entry in bundle.thermal_blocks))
        self.assertTrue(all(entry["preflight"]["thermally_suspect"] for entry in bundle.thermal_blocks))
        self.assertTrue(all(entry["preflight"]["retried"] for entry in bundle.thermal_blocks))
        # Tagging, not refusing: every slot still ran (same count as the
        # nominal-thermal case) -- the block is annotated, never dropped.
        self.assertEqual(len(bundle.measurements.raw_records), 8)

    def test_thermal_gate_policy_refuse_skips_suspect_block_samples(self) -> None:
        """"refuse" policy: a persistently throttled block's samples are never taken."""

        preflight, calls = self._counting_preflight(thermally_suspect=True)
        policy = EvaluationPolicy(
            warmup_runs=1, measurement_runs=2, timeout_seconds=1, max_output_bytes=4096, thermal_gate_policy="refuse",
        )
        bundle = self._evaluator(policy=policy, thermal_preflight=preflight).evaluate(self.proposal)
        # The preflight still runs for every block -- refusal is decided per
        # block, not by short-circuiting the whole evaluate() call.
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(entry["refused"] for entry in bundle.thermal_blocks))
        self.assertTrue(all(entry["policy"] == "refuse" for entry in bundle.thermal_blocks))
        # No samples were taken for either block -- the evaluator loop
        # `continue`s past a refused block's slots entirely.
        self.assertEqual(len(bundle.measurements.raw_records), 0)
        self.assertTrue(any(reason.startswith("missing:") for reason in bundle.measurements.rejection_reasons))
        self.assertFalse(bundle.accepted)

    def test_thermal_gate_only_refuses_the_specific_block_that_is_suspect(self) -> None:
        """Per-block gating: one throttled block among several is refused; the rest are not."""

        # Block 1's preflight makes one prober() call (nominal, no retry).
        # Block 2's preflight makes two (throttled, then throttled again on
        # retry) -- so the underlying prober must keep returning "throttled"
        # once it starts, not just once.
        nominal = ThermalReading("nominal", 100, None, None, "")
        throttled = ThermalReading("throttled", 40, None, None, "")
        prober_calls: list[None] = []

        def prober() -> ThermalReading:
            prober_calls.append(None)
            return nominal if len(prober_calls) == 1 else throttled

        calls: list[None] = []

        def preflight():
            calls.append(None)
            return _real_thermal_preflight(prober=prober, sleep=lambda seconds: None, retry_pause_seconds=30.0)

        policy = EvaluationPolicy(
            warmup_runs=1, measurement_runs=2, timeout_seconds=1, max_output_bytes=4096, thermal_gate_policy="refuse",
        )
        bundle = self._evaluator(policy=policy, thermal_preflight=preflight).evaluate(self.proposal)
        self.assertEqual(len(calls), 2)
        self.assertEqual([entry["refused"] for entry in bundle.thermal_blocks], [False, True])
        # Only the second block's four slots (2 baseline + 2 candidate) were skipped.
        self.assertEqual(len(bundle.measurements.raw_records), 4)
        self.assertTrue(all(record.arm is not None for record in bundle.measurements.raw_records))

    def test_evaluate_never_reads_external_isolation_metadata(self) -> None:
        bundle = self._evaluator(
            provider=ExplodingFixtureProvider(),
            authority=ExplodingFixtureAuthority(),
        ).evaluate(self.proposal)
        self.assertFalse(bundle.accepted)
        self.assertEqual(
            (
                bundle.isolation_provider_id,
                bundle.isolation_identity,
                bundle.isolation_verifier_id,
                bundle.isolation_verifier_identity,
                bundle.isolation_requirements,
            ),
            (None, None, None, None, None),
        )
        self.assertIn("production_isolation_unavailable", bundle.measurements.rejection_reasons)

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


@unittest.skipUnless(
    local_sandbox_primitives_available(),
    "local sandbox-exec primitives (macOS + sandbox-exec) are unavailable on this host",
)
class EvaluatorLocalSandboxTests(unittest.TestCase):
    """End-to-end: Evaluator -> execute_plan -> LocalSandboxProvider/Authority."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.workload = FrozenWorkload("evaluator-sandbox-fixture", knobs=(Knob("mode", "enum", values=("safe",)),))
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

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_real_local_sandbox_evaluation_is_accepted_and_promotion_eligible(self) -> None:
        evaluator = Evaluator(
            self.registry,
            baseline_runner_id="baseline",
            candidate_runner_id="candidate",
            oracle=ExactOutputOracle(b"ok\n"),
            artifact_root=str(self.root),
            policy=EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=5, max_output_bytes=4096),
            execution_policy=ExecutionPolicy(
                timeout_seconds=5,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                max_output_bytes=4096,
            ),
            provider=LocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        bundle = evaluator.evaluate(self.proposal)
        self.assertTrue(all(record.status is ExecutionStatus.SUCCESS for record in bundle.raw_records))
        self.assertTrue(all(record.stdout == b"ok\n" for record in bundle.raw_records))
        # Isolation identity is threaded from real execution evidence, not
        # hardcoded None: it must reflect the LocalSandboxProvider/Authority
        # actually used.
        self.assertEqual(bundle.isolation_provider_id, "local-sandbox-exec")
        self.assertEqual(bundle.isolation_verifier_id, "local-sandbox-authority")
        self.assertIsNotNone(bundle.isolation_identity)
        self.assertIsNotNone(bundle.isolation_verifier_identity)
        self.assertEqual(bundle.isolation_requirements, frozenset({"network_denial", "descendant_containment"}))
        self.assertNotIn("production_isolation_unavailable", bundle.measurements.rejection_reasons)
        self.assertTrue(bundle.accepted)
        self.assertTrue(bundle.promotion_eligible)
        self.assertTrue(bundle.measurements.accepted)
        self.assertTrue(bundle.measurements.promotion_eligible)

    def test_evaluate_never_reads_local_sandbox_provider_or_authority_identity_directly(self) -> None:
        # Same invariant as test_evaluate_never_reads_external_isolation_metadata,
        # exercised against the real local sandbox provider/authority rather
        # than a fixture: Evaluator must derive bundle-level isolation
        # identity only from what real execution actually returned.
        class ExplodingLocalSandboxProvider(LocalSandboxProvider):
            @property
            def provider_id(self):
                raise AssertionError("evaluator must not read provider_id")

            @property
            def identity(self):
                raise AssertionError("evaluator must not read provider identity")

        evaluator = Evaluator(
            self.registry,
            baseline_runner_id="baseline",
            candidate_runner_id="candidate",
            oracle=ExactOutputOracle(b"ok\n"),
            artifact_root=str(self.root),
            policy=EvaluationPolicy(warmup_runs=1, measurement_runs=1, timeout_seconds=5, max_output_bytes=4096),
            execution_policy=ExecutionPolicy(
                timeout_seconds=5,
                max_stdout_bytes=4096,
                max_stderr_bytes=4096,
                max_output_bytes=4096,
            ),
            provider=ExplodingLocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        bundle = evaluator.evaluate(self.proposal)
        self.assertFalse(bundle.accepted)
        self.assertIsNone(bundle.isolation_provider_id)
        self.assertIsNone(bundle.isolation_identity)


if __name__ == "__main__":
    unittest.main()
