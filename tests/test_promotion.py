from __future__ import annotations

import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity
from auto_mlx.errors import ContractError, FailureCode
from auto_mlx.promotion import ACTIVATE, NATIVE, PromotionDecision, activate, make_promotion_decision, rollback
from auto_mlx.receipts import (
    CLAIMS_WITHHELD,
    ContentAddressedStore,
    RawSample,
    Receipt,
    ReceiptValidation,
    receipt_attestation,
    validate_receipt,
)


class PromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = FrozenWorkload("promotion-test", knobs=(Knob("mode", "enum", values=("eager",)),))
        self.candidate = CandidateProposal("grid", self.workload, {"mode": "eager"})
        self.policy = EvaluationPolicy(warmup_runs=0, measurement_runs=2)
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.receipt = Receipt(
            self.workload,
            self.candidate,
            self.policy,
            self.runtime,
            (
                RawSample(0, 100, 120, "ok", "ok", 0),
                RawSample(1, 110, 130, "ok", "ok", 0),
            ),
            created_at_ns=100,
        )
        self.key = b"supervisor-key-for-tests"

    def validation(self, artifact_root: str):
        tag = receipt_attestation(self.receipt, self.key)
        return validate_receipt(self.receipt, artifact_root=artifact_root, attestation=tag, attestation_key=self.key)

    def test_only_complete_validated_local_receipt_can_activate(self) -> None:
        with tempfile.TemporaryDirectory() as artifact_root:
            validation = self.validation(artifact_root)
            decision = make_promotion_decision(validation, now_ns=110, attestation_key=self.key)
        self.assertEqual(decision.action, ACTIVATE)
        self.assertEqual(decision.claims, {"public": CLAIMS_WITHHELD, "performance": CLAIMS_WITHHELD})
        self.assertEqual(PromotionDecision.from_dict(decision.to_dict()), decision)

        with self.assertRaises(ContractError) as context:
            make_promotion_decision(self.receipt)  # type: ignore[arg-type]
        self.assertEqual(context.exception.code, FailureCode.PROMOTION_REJECTED)

    def test_invalid_receipt_and_evaluator_like_claims_never_self_promote(self) -> None:
        data = self.receipt.to_dict()
        data["metrics"]["gain"]["candidate_sum_ns"] = 1
        invalid = validate_receipt(data)
        decision = make_promotion_decision(invalid, now_ns=110)
        self.assertEqual(decision.action, NATIVE)
        self.assertNotEqual(decision.reason, "complete_validated_local_receipt")
        with self.assertRaises(ContractError):
            PromotionDecision(
                source="evaluator_recommendation",
                action=ACTIVATE,
                reason="recommended",
                receipt_id=self.receipt.receipt_id,
                workload=self.workload,
                candidate=self.candidate,
                policy=self.policy,
                runtime=self.runtime,
                artifacts=(),
            )

    def test_activation_is_persisted_and_rollback_only_moves_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            store.put_receipt(self.receipt)
            validation = self.validation(raw_root)
            decision = activate(store, validation, artifact_root=raw_root, attestation_key=self.key, now_ns=110)
            self.assertEqual(store.current_decision_id(), decision.decision_id)
            self.assertEqual(store.get_decision(decision.decision_id), decision.to_dict())
            rolled_back = rollback(store, now_ns=120)
            self.assertEqual(rolled_back.action, NATIVE)
            self.assertEqual(store.current_decision_id(), "native_fallback")
            self.assertEqual(store.get_decision(decision.decision_id), decision.to_dict())

    def test_claims_cannot_be_relabelled_as_public_or_performance_proof(self) -> None:
        validation = validate_receipt(self.receipt)
        decision = make_promotion_decision(validation)
        with self.assertRaises(ContractError):
            PromotionDecision.from_dict({**decision.to_dict(), "claims": {"public": "proven", "performance": "proven"}})

    def test_self_authored_receipt_without_supervisor_proof_clears_stale_activation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            store.put_receipt(self.receipt)
            active = activate(store, self.validation(raw_root), artifact_root=raw_root, attestation_key=self.key, now_ns=110)
            self.assertEqual(store.current_decision_id(), active.decision_id)
            rejected = activate(store, validate_receipt(self.receipt), now_ns=120)
            self.assertEqual(rejected.action, NATIVE)
            self.assertEqual(store.current_decision_id(), "native_fallback")

    def test_source_action_pairing_rejects_rollback_activation(self) -> None:
        with self.assertRaises(ContractError):
            PromotionDecision(
                source="rollback",
                action=ACTIVATE,
                reason="forged",
                receipt_id=self.receipt.receipt_id,
                workload=self.workload,
                candidate=self.candidate,
                policy=self.policy,
                runtime=self.runtime,
                attestation="0" * 64,
            )

    def test_activation_requires_artifact_root_and_proof(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            store.put_receipt(self.receipt)
            rejected = activate(store, validate_receipt(self.receipt), now_ns=110)
            self.assertEqual(rejected.action, NATIVE)
            self.assertEqual(store.current_decision_id(), "native_fallback")

    def test_signed_slower_receipt_is_storable_evidence_but_never_activates(self) -> None:
        slower = Receipt(
            self.workload,
            self.candidate,
            self.policy,
            self.runtime,
            (
                RawSample(0, 150, 100, "ok", "ok", 0),
                RawSample(1, 160, 110, "ok", "ok", 0),
            ),
            created_at_ns=100,
        )
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            store.put_receipt(slower)
            tag = receipt_attestation(slower, self.key)
            validation = validate_receipt(slower, artifact_root=raw_root, attestation=tag, attestation_key=self.key)
            self.assertTrue(validation.ok)
            self.assertFalse(validation.recomputed["metrics"]["gain"]["improved"])
            decision = activate(store, validation, artifact_root=raw_root, attestation_key=self.key, now_ns=110)
            self.assertEqual(decision.action, NATIVE)
            self.assertEqual(decision.reason, "gain_not_positive")
            self.assertEqual(store.current_decision_id(), "native_fallback")

    def test_promotion_recomputes_hmac_instead_of_trusting_validation_marker(self) -> None:
        with tempfile.TemporaryDirectory() as artifact_root:
            validation = self.validation(artifact_root)
            self.assertEqual(
                make_promotion_decision(validation, attestation_key=b"wrong").action,
                NATIVE,
            )
            self.assertEqual(
                make_promotion_decision(validation, attestation_key=self.key).action,
                ACTIVATE,
            )

    def test_forged_validation_flags_without_internal_proof_cannot_promote(self) -> None:
        with tempfile.TemporaryDirectory() as artifact_root:
            tag = receipt_attestation(self.receipt, self.key)
            forged = ReceiptValidation(
                self.receipt,
                True,
                True,
                True,
                (),
                {},
                attested=True,
                attestation=tag,
                artifacts_verified=True,
            )
            self.assertFalse(forged.ok)
            self.assertEqual(make_promotion_decision(forged, attestation_key=self.key).action, NATIVE)

    def test_activation_requires_exact_receipt_in_immutable_store(self) -> None:
        with tempfile.TemporaryDirectory() as store_root, tempfile.TemporaryDirectory() as artifact_root:
            store = ContentAddressedStore(store_root)
            rejected = activate(
                store,
                self.validation(artifact_root),
                artifact_root=artifact_root,
                attestation_key=self.key,
                now_ns=110,
            )
            self.assertEqual(rejected.action, NATIVE)
            self.assertEqual(store.current_decision_id(), "native_fallback")

    def test_fsync_failure_surfaces_and_leaves_native_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root, tempfile.TemporaryDirectory() as artifact_root:
            store = ContentAddressedStore(raw_root)
            store.put_receipt(self.receipt)
            with patch("auto_mlx.receipts.os.fsync", side_effect=OSError(5, "I/O failure")):
                with self.assertRaises(ContractError) as context:
                    activate(store, self.validation(artifact_root), artifact_root=artifact_root, attestation_key=self.key, now_ns=110)
            self.assertEqual(context.exception.code, FailureCode.RUNTIME_FAILURE)
            self.assertEqual(store.current_decision_id(), "native_fallback")


if __name__ == "__main__":
    unittest.main()
