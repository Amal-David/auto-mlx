from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity, sha256_hex
from auto_mlx.errors import ContractError, Failure, FailureCode, UnknownFieldError
from auto_mlx.evaluator import Evaluator, Observation, ObservationBundle, _execution_policy_digest, _execution_policy_from_contract
from auto_mlx.executor import ExecutionRecord, ExecutionStatus, IsolationAuthority, IsolationClaim, IsolatedProcess, IsolationProvider, TrustedRunner, TrustedRunnerRegistry
from auto_mlx.measurement import MeasurementSample, PairedMeasurementPlan, assemble_measurement_bundle
from auto_mlx.oracle import ExactOutputOracle
from auto_mlx.promotion import ACTIVATE, make_promotion_decision, rollback
from auto_mlx.receipts import (
    MAX_CURRENT_POINTER_BYTES,
    MAX_STORED_DECISION_BYTES,
    MAX_STORED_RECEIPT_BYTES,
    ContentAddressedStore,
    RawSample,
    Receipt,
    receipt_attestation,
    validate_receipt,
)


class _ReceiptIsolationProvider(IsolationProvider):
    def __init__(self) -> None:
        super().__init__("receipt-test-isolation", "e" * 64)

    def enforce(self, argv, **kwargs):
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
        return IsolatedProcess(process, self._claim("d" * 64))


class _ReceiptTestAuthority(IsolationAuthority):
    def __init__(self) -> None:
        super().__init__("receipt-test-verifier", "f" * 64, production_eligible=False)

    def verify(self, provider, process, claim: IsolationClaim):
        return self._attest(provider, claim)


class ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = FrozenWorkload("receipt-test", knobs=(Knob("mode", "enum", values=("eager",)),))
        self.candidate = {"provider_id": "grid", "workload_hash": self.workload.workload_hash, "config": {"mode": "eager"}}
        from auto_mlx import CandidateProposal

        self.candidate = CandidateProposal.from_dict(
            {
                **self.candidate,
                "candidate_id": sha256_hex(self.candidate),
            },
            self.workload,
        )
        self.policy = EvaluationPolicy(warmup_runs=1, measurement_runs=2)
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.key = b"supervisor-key-for-tests"

    def receipt(self, *, mismatch: bool = False, samples: int = 2) -> Receipt:
        raw = [
            RawSample(index, 100 + index * 10, 120 + index * 10, "wrong" if mismatch and index == 1 else "ok", "ok", index)
            for index in range(samples)
        ]
        return Receipt(self.workload, self.candidate, self.policy, self.runtime, raw, created_at_ns=100)

    def test_identity_and_all_derived_fields_are_recomputed_from_raw_samples(self) -> None:
        receipt = self.receipt()
        body = receipt.to_dict()
        body.pop("receipt_id")
        self.assertEqual(receipt.receipt_id, sha256_hex(body))
        self.assertEqual(receipt.aggregates["candidate"]["sum_ns"], 210)
        self.assertEqual(receipt.metrics["gain"]["delta_ns"], 40)
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = str(Path(raw_root).resolve())
            tag = receipt_attestation(receipt, self.key)
            self.assertTrue(
                validate_receipt(receipt, artifact_root=artifact_root, attestation=tag, attestation_key=self.key).ok
            )

        tampered = receipt.to_dict()
        tampered["aggregates"]["candidate"]["sum_ns"] = 999
        result = validate_receipt(tampered)
        self.assertFalse(result.valid)
        self.assertTrue(any(f.code == FailureCode.IDENTITY_MISMATCH for f in result.failures))

    def test_oracle_mismatch_and_incomplete_samples_are_classified(self) -> None:
        mismatch = validate_receipt(self.receipt(mismatch=True))
        self.assertFalse(mismatch.ok)
        self.assertTrue(any(f.code == FailureCode.ORACLE_MISMATCH for f in mismatch.failures))
        incomplete = validate_receipt(self.receipt(samples=1))
        self.assertFalse(incomplete.complete)
        self.assertTrue(any(f.code == FailureCode.INVALID_VALUE for f in incomplete.failures))

    def test_cherry_picked_sample_indices_are_rejected(self) -> None:
        cherry_picked = Receipt(
            self.workload,
            self.candidate,
            self.policy,
            self.runtime,
            (RawSample(0, 100, 120, "ok", "ok", 0), RawSample(2, 110, 130, "ok", "ok", 0)),
            created_at_ns=100,
        )
        result = validate_receipt(cherry_picked)
        self.assertFalse(result.ok)
        self.assertTrue(any(f.code == FailureCode.INVALID_VALUE for f in result.failures))

    def test_nested_raw_evidence_is_deeply_immutable(self) -> None:
        source = {"tokens": [1, 2]}
        sample = RawSample(0, 1, 2, source, source, 0)
        source["tokens"].append(3)
        self.assertEqual(sample.actual_output["tokens"], (1, 2))
        with self.assertRaises(TypeError):
            sample.actual_output["tokens"] = ()  # type: ignore[index]
        receipt = self.receipt()
        with self.assertRaises(TypeError):
            receipt.aggregates["candidate"] = {}  # type: ignore[index]

    def test_closed_nested_wire_shapes_reject_child_metrics(self) -> None:
        data = self.receipt().to_dict()
        data["metrics"]["child_aggregate"] = {"sum_ns": 1}
        with self.assertRaises(UnknownFieldError):
            Receipt.from_dict(data)

    def test_failed_receipts_are_immutable_evidence_but_cannot_validate(self) -> None:
        failed = Receipt(
            self.workload,
            self.candidate,
            self.policy,
            self.runtime,
            (),
            created_at_ns=100,
            failure=Failure(FailureCode.TIMEOUT, "measurement timed out"),
            status="failed",
        )
        result = validate_receipt(failed)
        self.assertFalse(result.ok)
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            path = store.put_receipt(failed)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                store.put_receipt(failed)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(store.get_receipt(failed.receipt_id), failed)

    def test_current_receipt_pointer_is_atomic_and_content_addressed(self) -> None:
        receipt = self.receipt()
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            store.put_receipt(receipt)
            store.set_current_receipt(receipt.receipt_id)
            self.assertEqual(store.current_receipt_id(), receipt.receipt_id)

    def test_storage_root_symlink_is_rejected(self) -> None:
        receipt = self.receipt()
        with tempfile.TemporaryDirectory() as raw_root:
            base = Path(raw_root)
            target = base / "target"
            target.mkdir()
            alias = base / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(OSError):
                ContentAddressedStore(alias).put_receipt(receipt)

    def test_malformed_receipt_is_returned_as_invalid_validation(self) -> None:
        result = validate_receipt({"schema": "wrong"})
        self.assertFalse(result.valid)
        self.assertEqual(result.failures[0].code, FailureCode.INVALID_VALUE)

    def test_zero_timings_cannot_become_valid_evidence(self) -> None:
        receipt = Receipt(
            self.workload,
            self.candidate,
            self.policy,
            self.runtime,
            (RawSample(0, 0, 120, "ok", "ok", 0), RawSample(1, 110, 130, "ok", "ok", 0)),
            created_at_ns=100,
        )
        result = validate_receipt(receipt)
        self.assertFalse(result.valid)
        self.assertTrue(any(f.code == FailureCode.RUNTIME_FAILURE for f in result.failures))

    def test_fsync_unsupported_is_reported_as_non_durable_observation_storage(self) -> None:
        receipt = self.receipt()
        with tempfile.TemporaryDirectory() as raw_root, patch("auto_mlx.receipts.os.fsync", side_effect=OSError(95, "unsupported")):
            store = ContentAddressedStore(raw_root)
            store.put_receipt(receipt)
            self.assertFalse(store.last_durable)
            self.assertEqual(store.get_receipt(receipt.receipt_id), receipt)

    def test_stored_receipt_and_pointer_reads_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            receipt_path = store.put_receipt(self.receipt())
            receipt_path.write_bytes(b"x" * (MAX_STORED_RECEIPT_BYTES + 1))
            with self.assertRaises(ContractError):
                store.get_receipt(self.receipt().receipt_id)

            decision = rollback(store)
            decision_path = Path(raw_root) / "decisions" / f"{decision.decision_id}.json"
            decision_path.write_bytes(b"x" * (MAX_STORED_DECISION_BYTES + 1))
            with self.assertRaises(ContractError):
                store.get_decision(decision.decision_id)

            pointer_path = Path(raw_root) / "pointers" / "current_decision"
            pointer_path.write_bytes(b"x" * (MAX_CURRENT_POINTER_BYTES + 1))
            with self.assertRaises(ContractError):
                store.current_decision_id()

    def test_root_identity_is_checked_after_descriptor_walk(self) -> None:
        receipt = self.receipt()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root) / "store"
            root.mkdir()
            import auto_mlx.receipts as receipts_module
            original_open = receipts_module.os.open
            replacement = Path(raw_root) / "replacement"
            replacement.mkdir()
            moved = Path(raw_root) / "moved"
            swapped = False

            def swap_before_final_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if not swapped and path == root.name and kwargs.get("dir_fd") is not None:
                    swapped = True
                    root.rename(moved)
                    replacement.rename(root)
                return original_open(path, flags, *args, **kwargs)

            with patch("auto_mlx.receipts.os.open", side_effect=swap_before_final_open):
                with self.assertRaises(OSError):
                    ContentAddressedStore(root).put_receipt(receipt)

    def test_actual_observation_bundle_adapter_retains_four_slot_blocks_and_provenance(self) -> None:
        oracle = ExactOutputOracle(b"ok\n", label="receipt-oracle")
        provider = _ReceiptIsolationProvider()
        authority = _ReceiptTestAuthority()
        plan = PairedMeasurementPlan.create(
            self.policy.measurement_runs,
            candidate_id=self.candidate.candidate_id,
            workload_hash=self.workload.workload_hash,
            baseline_runner_id="baseline",
            baseline_runner_digest="1" * 64,
            candidate_runner_id="candidate",
            candidate_runner_digest="2" * 64,
            oracle=oracle,
            isolation_provider_id=provider.provider_id,
            isolation_identity=provider.identity,
            isolation_verifier_id=authority.verifier_id,
            isolation_verifier_identity=authority.identity,
            isolation_requirements=frozenset({"network_denial", "descendant_containment"}),
        )

        def record(sample_id: str, arm: str, elapsed: int) -> ExecutionRecord:
            runner_id = "baseline" if arm == "baseline" else "candidate"
            runner_digest = "1" * 64 if arm == "baseline" else "2" * 64
            return ExecutionRecord(
                candidate_id=self.candidate.candidate_id,
                workload_hash=self.workload.workload_hash,
                runner_id=runner_id,
                runner_digest=runner_digest,
                status=ExecutionStatus.SUCCESS,
                parent_elapsed_ns=elapsed,
                observation_id=sample_id,
                arm=arm,
                returncode=0,
                stdout=b"ok\n",
                isolation=authority._attest(provider, provider._claim("d" * 64)),
            )

        warmups = tuple(
            Observation(
                f"warmup-0001-{arm}",
                arm,
                record(f"warmup-0001-{arm}", arm, 50),
                oracle.evaluate(b"ok\n"),
            )
            for arm in ("baseline", "candidate")
        )
        samples = []
        for block in plan.blocks:
            for slot in block.slots:
                current = record(slot.sample_id, slot.arm, 100 + slot.slot_index + block.block_index * 10)
                samples.append(MeasurementSample(slot.sample_id, slot.block_id, slot.slot_index, slot.arm, current, oracle.evaluate(current.stdout)))
        measurements = assemble_measurement_bundle(plan, samples)
        bundle = ObservationBundle(
            candidate_id=self.candidate.candidate_id,
            workload_hash=self.workload.workload_hash,
            runtime=self.runtime,
            baseline_runner_id="baseline",
            baseline_runner_digest="1" * 64,
            candidate_runner_id="candidate",
            candidate_runner_digest="2" * 64,
            isolation_provider_id=provider.provider_id,
            isolation_identity=provider.identity,
            isolation_verifier_id=authority.verifier_id,
            isolation_verifier_identity=authority.identity,
            isolation_requirements=frozenset({"network_denial", "descendant_containment"}),
            warmups=warmups,
            measurements=measurements,
            policy_digest=sha256_hex(self.policy.to_dict()),
            execution_policy_digest=_execution_policy_digest(_execution_policy_from_contract(self.policy)),
            measurement_block_count=self.policy.measurement_runs,
            evaluation_policy=self.policy,
            execution_policy=_execution_policy_from_contract(self.policy),
            oracle=oracle,
            oracle_descriptor=oracle.descriptor,
        )
        # Every warmup and sample here carries real, matched isolation
        # evidence and a matching oracle, so the G1 evidence layer correctly
        # accepts this bundle now that acceptance is evidence-based rather
        # than an unconditional hold. receipts.py's status is now COMPUTED
        # from that same real evidence (bundle.accepted and
        # bundle.promotion_eligible), so this well-formed bundle now
        # produces a "complete" receipt with no failure -- the previous
        # unconditional status="failed" hold is gone. G2 promotion
        # activation is still a SEPARATE, independent gate: this test's own
        # validate_receipt() call below never supplies artifact_root, so
        # validation.artifacts_verified stays False and make_promotion_decision
        # still cannot ACTIVATE, exactly as asserted before -- that
        # assertion is unchanged and still exercises the real gate.
        self.assertTrue(bundle.accepted)
        self.assertTrue(bundle.promotion_eligible)
        receipt = Receipt.from_observation_bundle(
            bundle,
            self.workload,
            self.candidate,
            self.policy,
            oracle=oracle,
            created_at_ns=100,
        )
        self.assertEqual(receipt.status, "complete")
        self.assertIsNone(receipt.failure)
        validation = validate_receipt(
            receipt,
            attestation=receipt_attestation(receipt, self.key),
            attestation_key=self.key,
        )
        decision = make_promotion_decision(
            validation,
            now_ns=110,
            attestation_key=self.key,
        )
        self.assertNotEqual(decision.action, ACTIVATE)

    def test_actual_evaluator_slower_candidate_preserves_signed_regression(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            script = root / "runner.py"
            script.write_text(
                "import sys, time\n"
                "if sys.argv[1] == 'slow': time.sleep(0.05)\n"
                "print('ok')\n",
                encoding="utf-8",
            )
            python_artifact = str(Path(sys.executable).resolve())
            baseline = TrustedRunner.from_command(
                "baseline",
                (sys.executable, str(script), "fast"),
                artifact_paths=(str(script), python_artifact),
            )
            candidate = TrustedRunner.from_command(
                "candidate",
                (sys.executable, str(script), "slow"),
                artifact_paths=(str(script), python_artifact),
            )
            workload = FrozenWorkload("slower-evaluator", knobs=(Knob("mode", "enum", values=("safe",)),))
            proposal = CandidateProposal("fixture", workload, {"mode": "safe"})
            policy = EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=2, max_output_bytes=4096)
            oracle = ExactOutputOracle(b"ok\n")
            evaluator = Evaluator(
                TrustedRunnerRegistry((baseline, candidate)),
                baseline_runner_id="baseline",
                candidate_runner_id="candidate",
                oracle=oracle,
                artifact_root=raw_root,
                policy=policy,
                provider=_ReceiptIsolationProvider(),
                authority=_ReceiptTestAuthority(),
            )
            bundle = evaluator.evaluate(proposal)
            self.assertFalse(bundle.accepted)
            with self.assertRaises(ContractError) as mismatched_policy:
                Receipt.from_observation_bundle(
                    bundle,
                    workload,
                    proposal,
                    EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=2, max_output_bytes=4095),
                    oracle=oracle,
                    created_at_ns=100,
                )
            self.assertEqual(mismatched_policy.exception.code, FailureCode.INVALID_POLICY)
            with self.assertRaises(ContractError) as forged_digest:
                Receipt.from_observation_bundle(
                    replace(bundle, policy_digest="0" * 64),
                    workload,
                    proposal,
                    policy,
                    oracle=oracle,
                    created_at_ns=100,
                )
            self.assertEqual(forged_digest.exception.code, FailureCode.INVALID_POLICY)
            receipt = Receipt.from_observation_bundle(bundle, workload, proposal, policy, oracle=oracle, created_at_ns=100)
            self.assertEqual(receipt.status, "failed")

    def test_actual_evaluator_faster_candidate_preserves_positive_regression(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            script = root / "runner.py"
            script.write_text(
                "import sys, time\n"
                "if sys.argv[1] == 'slow': time.sleep(0.05)\n"
                "print('ok')\n",
                encoding="utf-8",
            )
            python_artifact = str(Path(sys.executable).resolve())
            baseline = TrustedRunner.from_command(
                "baseline",
                (sys.executable, str(script), "slow"),
                artifact_paths=(str(script), python_artifact),
            )
            candidate = TrustedRunner.from_command(
                "candidate",
                (sys.executable, str(script), "fast"),
                artifact_paths=(str(script), python_artifact),
            )
            workload = FrozenWorkload("faster-evaluator", knobs=(Knob("mode", "enum", values=("safe",)),))
            proposal = CandidateProposal("fixture", workload, {"mode": "safe"})
            policy = EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=2, max_output_bytes=4096)
            oracle = ExactOutputOracle(b"ok\n")
            evaluator = Evaluator(
                TrustedRunnerRegistry((baseline, candidate)),
                baseline_runner_id="baseline",
                candidate_runner_id="candidate",
                oracle=oracle,
                artifact_root=raw_root,
                policy=policy,
                provider=_ReceiptIsolationProvider(),
                authority=_ReceiptTestAuthority(),
            )
            bundle = evaluator.evaluate(proposal)
            self.assertFalse(bundle.accepted)
            receipt = Receipt.from_observation_bundle(bundle, workload, proposal, policy, oracle=oracle, created_at_ns=100)
            self.assertEqual(receipt.status, "failed")


if __name__ == "__main__":
    unittest.main()
