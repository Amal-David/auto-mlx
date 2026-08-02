from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx.errors import ContractError
from auto_mlx.executor import ExecutionRecord, ExecutionStatus, IsolationAuthority, IsolationClaim, IsolationProvider
from auto_mlx.measurement import (
    MeasurementRejected,
    MeasurementSample,
    PairedMeasurementPlan,
    assemble_measurement_bundle,
)
from auto_mlx.oracle import ExactOutputOracle


class TestIsolationProvider(IsolationProvider):
    def __init__(self) -> None:
        super().__init__("measurement-test", "3" * 64)

    def enforce(self, argv, **kwargs):
        raise RuntimeError("not used")


class TestOnlyIsolationAuthority(IsolationAuthority):
    def __init__(self) -> None:
        super().__init__("measurement-test-verifier", "6" * 64, production_eligible=False)

    def verify(self, provider, process, claim: IsolationClaim):
        return self._attest(provider, claim)


class MeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate_id = "c" * 64
        self.workload_hash = "d" * 64
        self.baseline_runner_id = "baseline"
        self.baseline_runner_digest = "4" * 64
        self.candidate_runner_id = "candidate"
        self.candidate_runner_digest = "5" * 64
        self.oracle = ExactOutputOracle(b"ok\n")
        self.provider = TestIsolationProvider()
        self.authority = TestOnlyIsolationAuthority()
        self.plan = PairedMeasurementPlan.create(
            2,
            candidate_id=self.candidate_id,
            workload_hash=self.workload_hash,
            baseline_runner_id=self.baseline_runner_id,
            baseline_runner_digest=self.baseline_runner_digest,
            candidate_runner_id=self.candidate_runner_id,
            candidate_runner_digest=self.candidate_runner_digest,
            oracle=self.oracle,
            isolation_provider_id=self.provider.provider_id,
            isolation_identity=self.provider.identity,
            isolation_verifier_id=self.authority.verifier_id,
            isolation_verifier_identity=self.authority.identity,
            isolation_requirements=frozenset({"network_denial", "descendant_containment"}),
        )

    def _sample(self, slot, *, status: ExecutionStatus = ExecutionStatus.SUCCESS, elapsed: int = 100, output: bytes = b"ok\n", isolation=True):
        runner_id = self.baseline_runner_id if slot.arm == "baseline" else self.candidate_runner_id
        runner_digest = self.baseline_runner_digest if slot.arm == "baseline" else self.candidate_runner_digest
        record = ExecutionRecord(
            candidate_id=self.candidate_id,
            workload_hash=self.workload_hash,
            runner_id=runner_id,
            runner_digest=runner_digest,
            status=status,
            parent_elapsed_ns=elapsed,
            observation_id=slot.sample_id,
            arm=slot.arm,
            returncode=0 if status is ExecutionStatus.SUCCESS else 1,
            stdout=output,
            isolation=self.authority._attest(self.provider, self.provider._claim("a" * 64)) if isolation else None,
        )
        return MeasurementSample(slot.sample_id, slot.block_id, slot.slot_index, slot.arm, record, self.oracle.evaluate(output))

    def test_plan_is_complete_and_deterministic_abba_then_baab(self) -> None:
        self.assertEqual(
            [block.sequence for block in self.plan.blocks],
            [
                ("baseline", "candidate", "candidate", "baseline"),
                ("candidate", "baseline", "baseline", "candidate"),
            ],
        )
        self.assertEqual(len(self.plan.expected_sample_ids), 8)
        self.assertEqual(self.plan.plan_digest, PairedMeasurementPlan.create(
            2,
            candidate_id=self.candidate_id,
            workload_hash=self.workload_hash,
            baseline_runner_id=self.baseline_runner_id,
            baseline_runner_digest=self.baseline_runner_digest,
            candidate_runner_id=self.candidate_runner_id,
            candidate_runner_digest=self.candidate_runner_digest,
            oracle=self.oracle,
            isolation_provider_id=self.provider.provider_id,
            isolation_identity=self.provider.identity,
            isolation_verifier_id=self.authority.verifier_id,
            isolation_verifier_identity=self.authority.identity,
            isolation_requirements=frozenset({"network_denial", "descendant_containment"}),
        ).plan_digest)

    def test_complete_bundle_retains_raw_records_and_drift_inputs(self) -> None:
        samples = [
            self._sample(slot, elapsed=100 + block.block_index * 10 + index)
            for block in self.plan.blocks
            for index, slot in enumerate(block.slots)
        ]
        bundle = assemble_measurement_bundle(self.plan, samples)
        self.assertTrue(bundle.accepted)
        self.assertEqual(len(bundle.raw_samples), 8)
        self.assertEqual(len(bundle.raw_records), 8)
        self.assertEqual(bundle.blocks[0].dispersion_inputs.ordered_parent_elapsed_ns, (100, 101, 102, 103))
        self.assertEqual(bundle.blocks[0].baseline_drift_ns, 3)
        self.assertEqual(bundle.blocks[1].dispersion_inputs.candidate_elapsed_ns, (110, 113))

    def test_missing_slot_is_rejected_and_retained_as_placeholder(self) -> None:
        samples = [self._sample(slot) for slot in self.plan.blocks[0].slots]
        samples.extend(self._sample(slot) for slot in self.plan.blocks[1].slots[:3])
        bundle = assemble_measurement_bundle(self.plan, samples)
        self.assertFalse(bundle.accepted)
        self.assertTrue(any(reason.startswith("missing:") for reason in bundle.rejection_reasons))
        self.assertEqual(len(bundle.raw_samples), 8)
        self.assertEqual(len(bundle.raw_records), 7)

    def test_failed_oracle_mismatched_and_forged_metadata_reject(self) -> None:
        samples = []
        for block in self.plan.blocks:
            for slot in block.slots:
                if slot.sample_id == self.plan.blocks[0].slots[0].sample_id:
                    samples.append(self._sample(slot, status=ExecutionStatus.EXIT_FAILURE))
                elif slot.sample_id == self.plan.blocks[1].slots[1].sample_id:
                    samples.append(self._sample(slot, output=b"wrong\n"))
                else:
                    samples.append(self._sample(slot))
        forged_slot = self.plan.blocks[0].slots[1]
        forged = self._sample(forged_slot)
        samples[samples.index(forged)] = MeasurementSample(
            forged.sample_id,
            forged.block_id,
            forged.slot_index,
            forged.arm,
            forged.record,
            self.oracle.evaluate(b"wrong\n"),
        )
        bundle = assemble_measurement_bundle(self.plan, samples)
        self.assertFalse(bundle.accepted)
        self.assertTrue(any("execution_failure" in reason for reason in bundle.rejection_reasons))
        self.assertTrue(any("oracle_mismatch" in reason for reason in bundle.rejection_reasons))
        self.assertTrue(any("forged_oracle_metadata" in reason for reason in bundle.rejection_reasons))
        self.assertEqual(len(bundle.raw_records), 8)

    def test_provenance_and_verified_isolation_are_required(self) -> None:
        slot = self.plan.blocks[0].slots[0]
        sample = self._sample(slot, isolation=False)
        bundle = assemble_measurement_bundle(self.plan, [
            sample if item.sample_id == slot.sample_id else self._sample(item)
            for block in self.plan.blocks
            for item in block.slots
        ])
        self.assertFalse(bundle.accepted)
        self.assertTrue(any("isolation_unverified" in reason for reason in bundle.rejection_reasons))

        bad_record = self._sample(slot).record
        object.__setattr__(bad_record, "observation_id", "wrong-slot")
        bad_sample = MeasurementSample(slot.sample_id, slot.block_id, slot.slot_index, slot.arm, bad_record, self.oracle.evaluate(b"ok\n"))
        full = [bad_sample if item.sample_id == slot.sample_id else self._sample(item) for block in self.plan.blocks for item in block.slots]
        bad_bundle = assemble_measurement_bundle(self.plan, full)
        self.assertFalse(bad_bundle.accepted)
        self.assertTrue(any("record_slot_identity_mismatch" in reason for reason in bad_bundle.rejection_reasons))

    def test_proof_from_another_provider_is_rejected(self) -> None:
        other = TestIsolationProvider()
        object.__setattr__(other, "_provider_id", "other-provider")
        slot = self.plan.blocks[0].slots[0]
        foreign_record = self._sample(slot).record
        object.__setattr__(foreign_record, "isolation", self.authority._attest(other, other._claim("b" * 64)))
        samples = [
            MeasurementSample(
                item.sample_id,
                item.block_id,
                item.slot_index,
                item.arm,
                foreign_record if item.sample_id == slot.sample_id else self._sample(item).record,
                self.oracle.evaluate(b"ok\n"),
            )
            for block in self.plan.blocks
            for item in block.slots
        ]
        bundle = assemble_measurement_bundle(self.plan, samples)
        self.assertFalse(bundle.accepted)
        self.assertTrue(any("isolation_provider_mismatch" in reason for reason in bundle.rejection_reasons))

    def test_success_with_nonzero_returncode_is_not_a_measurement_record(self) -> None:
        slot = self.plan.blocks[0].slots[0]
        with self.assertRaises(ContractError):
            self._sample(slot).record.__class__(
                candidate_id=self.candidate_id,
                workload_hash=self.workload_hash,
                runner_id=self.baseline_runner_id,
                runner_digest=self.baseline_runner_digest,
                status=ExecutionStatus.SUCCESS,
                parent_elapsed_ns=100,
                observation_id=slot.sample_id,
                arm=slot.arm,
                returncode=1,
                stdout=b"ok\n",
                isolation=self.authority._attest(self.provider, self.provider._claim("a" * 64)),
            )

    def test_unbound_and_cherry_picked_plans_are_rejected(self) -> None:
        unbound = PairedMeasurementPlan.create(1)
        slot = unbound.blocks[0].slots[0]
        bundle = assemble_measurement_bundle(unbound, [self._sample(slot)])
        self.assertFalse(bundle.accepted)
        self.assertIn("unbound_plan", bundle.rejection_reasons)

        samples = [self._sample(slot) for block in self.plan.blocks for slot in block.slots]
        samples.append(self._sample(self.plan.blocks[0].slots[0]))
        bundle = assemble_measurement_bundle(self.plan, samples)
        self.assertFalse(bundle.accepted)
        self.assertTrue(any("duplicate_sample_ids" in reason for reason in bundle.rejection_reasons))
        with self.assertRaises(MeasurementRejected):
            bundle.require_complete()


if __name__ == "__main__":
    unittest.main()
