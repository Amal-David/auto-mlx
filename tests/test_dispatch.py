from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import Artifact, CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity
from auto_mlx.canonical import MAX_JSON_DEPTH
from auto_mlx.dispatch import CANDIDATE_MODE, NATIVE_MODE, dispatch
from auto_mlx.promotion import activate, rollback
from auto_mlx.receipts import ContentAddressedStore, RawSample, Receipt, receipt_attestation, validate_receipt
from _wave_b_fixtures import build_evaluator_bundle_receipt


class DispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = FrozenWorkload("dispatch-test", knobs=(Knob("mode", "enum", values=("eager", "slow")),))
        self.candidate = CandidateProposal("grid", self.workload, {"mode": "eager"})
        # max_measurement_runs == measurement_runs: see test_promotion.py's
        # identical comment -- this fixture hand-builds exactly
        # measurement_runs blocks, so any verdict is a legitimate place to
        # have stopped.
        self.policy = EvaluationPolicy(warmup_runs=0, measurement_runs=2, max_measurement_runs=2)
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        # Wave B: dispatch now gates on the independently recomputed
        # statistics verdict too (see auto_mlx.dispatch), which only exists
        # for evaluator-bundle-backed receipts -- so this fixture builds a
        # genuinely decisive "improved" one (see _wave_b_fixtures) rather
        # than the older raw-sample lane, which is now correctly never
        # promotable.
        self.receipt = build_evaluator_bundle_receipt(self.workload, self.candidate, self.policy, self.runtime)
        self.key = b"supervisor-key-for-tests"

    def active_store(self) -> tuple[ContentAddressedStore, str, str]:
        raw_root = tempfile.TemporaryDirectory()
        self.addCleanup(raw_root.cleanup)
        artifact_root = str(Path(raw_root.name).resolve())
        store = ContentAddressedStore(artifact_root)
        store.put_receipt(self.receipt)
        tag = receipt_attestation(self.receipt, self.key)
        validation = validate_receipt(self.receipt, artifact_root=artifact_root, attestation=tag, attestation_key=self.key)
        decision = activate(store, validation, artifact_root=artifact_root, attestation_key=self.key, now_ns=110)
        return store, decision.decision_id, artifact_root

    def test_dispatch_requires_exact_workload_candidate_policy_runtime_and_artifacts(self) -> None:
        store, _, artifact_root = self.active_store()
        result = dispatch(store, self.workload, self.candidate, self.policy, self.runtime, artifact_root=artifact_root, attestation_key=self.key, now_ns=120, max_age_ns=100)
        self.assertEqual(result.mode, CANDIDATE_MODE)
        self.assertEqual(result.candidate_id, self.candidate.candidate_id)

        altered_workload = FrozenWorkload("dispatch-test-v2", knobs=self.workload.knobs)
        altered_candidate = CandidateProposal("grid", altered_workload, {"mode": "eager"})
        self.assertEqual(
            dispatch(store, altered_workload, altered_candidate, self.policy, self.runtime, artifact_root=artifact_root, attestation_key=self.key, now_ns=120, max_age_ns=100).mode,
            NATIVE_MODE,
        )
        altered_policy = EvaluationPolicy(warmup_runs=0, measurement_runs=1)
        self.assertEqual(
            dispatch(store, self.workload, self.candidate, altered_policy, self.runtime, artifact_root=artifact_root, attestation_key=self.key, now_ns=120, max_age_ns=100).mode,
            NATIVE_MODE,
        )
        altered_runtime = RuntimeIdentity("python", "3.12.0", "Darwin", "arm64")
        self.assertEqual(
            dispatch(store, self.workload, self.candidate, self.policy, altered_runtime, artifact_root=artifact_root, attestation_key=self.key, now_ns=120, max_age_ns=100).mode,
            NATIVE_MODE,
        )
        mismatched_artifact = Artifact("not-in-workload.bin", "0" * 64, 0)
        self.assertEqual(
            dispatch(
                store,
                self.workload,
                self.candidate,
                self.policy,
                self.runtime,
                artifacts=(mismatched_artifact,),
                artifact_root=artifact_root,
                attestation_key=self.key,
                now_ns=120,
                max_age_ns=100,
            ).mode,
            NATIVE_MODE,
        )

    def test_missing_stale_and_tampered_state_always_falls_back(self) -> None:
        # `.mode == NATIVE_MODE` alone would also be satisfied by an unrelated
        # rejection (for example a spuriously unresolved artifact_root), so
        # each case additionally asserts dispatch()'s specific `.reason`
        # string to confirm the intended mechanism -- staleness, a tampered
        # receipt, and an absent activation pointer -- is what actually fired.
        store, decision_id, artifact_root = self.active_store()
        stale = dispatch(store, self.workload, self.candidate, self.policy, self.runtime, artifact_root=artifact_root, attestation_key=self.key, now_ns=1_000, max_age_ns=10)
        self.assertEqual(stale.mode, NATIVE_MODE)
        self.assertEqual(stale.reason, "activation_stale")

        receipt_path = Path(store.root) / "receipts" / f"{self.receipt.receipt_id}.json"
        tampered = self.receipt.to_dict()
        tampered["raw_samples"][0]["duration_ns"] = 999
        # Writing with stdlib json.dumps (not canonical_json) means the tamper
        # is caught even earlier than the raw-sample recompute: get_receipt's
        # `canonical_bytes(parsed) != payload` check rejects the non-canonical
        # bytes on disk with IDENTITY_MISMATCH, which dispatch()'s outer
        # `except Exception` safety boundary reports as "dispatch_state_invalid".
        receipt_path.write_text(__import__("json").dumps(tampered), encoding="utf-8")
        tampered_result = dispatch(store, self.workload, self.candidate, self.policy, self.runtime, artifact_root=artifact_root, attestation_key=self.key, now_ns=120, max_age_ns=100)
        self.assertEqual(tampered_result.mode, NATIVE_MODE)
        self.assertEqual(tampered_result.reason, "dispatch_state_invalid")

        rollback(store, now_ns=130)
        missing = dispatch(store, self.workload, self.candidate, self.policy, self.runtime, now_ns=140, max_age_ns=100)
        self.assertEqual(missing.mode, NATIVE_MODE)
        self.assertEqual(missing.reason, "native_fallback_pointer")
        self.assertTrue(decision_id)

    def test_missing_pointer_is_native_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            result = dispatch(store, self.workload, self.candidate, self.policy, self.runtime)
            self.assertEqual(result.mode, NATIVE_MODE)

    def test_dispatch_wire_identity_is_closed(self) -> None:
        store, _, artifact_root = self.active_store()
        result = dispatch(store, self.workload, self.candidate, self.policy, self.runtime, artifact_root=artifact_root, attestation_key=self.key, now_ns=120, max_age_ns=100)
        self.assertEqual(type(result.from_dict(result.to_dict())).__name__, "DispatchResult")
        with self.assertRaises(Exception):
            result.from_dict({**result.to_dict(), "surprise": True})

    def test_missing_artifact_root_and_wrong_key_fall_back(self) -> None:
        store, _, artifact_root = self.active_store()
        self.assertEqual(dispatch(store, self.workload, self.candidate, self.policy, self.runtime, now_ns=120).mode, NATIVE_MODE)
        self.assertEqual(
            dispatch(store, self.workload, self.candidate, self.policy, self.runtime, artifact_root=artifact_root, attestation_key=b"wrong", now_ns=120).mode,
            NATIVE_MODE,
        )

    def test_pointer_flip_during_receipt_load_falls_back_native(self) -> None:
        store, _, artifact_root = self.active_store()
        real_get_receipt = store.get_receipt

        def flip_pointer(receipt_id: str):
            store.set_current_decision("native_fallback")
            return real_get_receipt(receipt_id)

        with patch.object(store, "get_receipt", side_effect=flip_pointer):
            result = dispatch(
                store,
                self.workload,
                self.candidate,
                self.policy,
                self.runtime,
                artifact_root=artifact_root,
                attestation_key=self.key,
                now_ns=120,
                max_age_ns=100,
            )
        self.assertEqual(result.mode, NATIVE_MODE)
        self.assertEqual(store.current_decision_id(), "native_fallback")

    def test_dispatch_mapping_ingress_rejects_overdeep_workload_before_state_use(self) -> None:
        nested: object = "leaf"
        for _ in range(MAX_JSON_DEPTH - 1):
            nested = [nested]
        deep_workload = FrozenWorkload("deep-dispatch", parameters={"nested": nested})
        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            result = dispatch(
                store,
                deep_workload.to_dict(),
                self.candidate.to_dict(),
                self.policy.to_dict(),
                self.runtime.to_dict(),
            )
        self.assertEqual(result.mode, NATIVE_MODE)


if __name__ == "__main__":
    unittest.main()
