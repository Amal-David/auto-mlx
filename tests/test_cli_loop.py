"""Tests for the real local evidence-gated CLI loop: evaluate, promote,
dispatch, rollback, and keys ensure (auto_mlx.cli).

Two groups, matching tests/test_supervisor.py's convention:

- ``CLILoopEndToEndTests`` (``skipUnless`` MLX + local sandbox primitives):
  the full honest chain through the real CLI surface -- evaluate -> promote
  -> dispatch -> dispatch --execute -> rollback -> dispatch -- against the
  real ``toy-matmul`` reference workload.  These run for real on any host
  with MLX and macOS ``sandbox-exec`` available.
- ``CLILoopFailClosedTests`` (no skip: runs on every host): the fail-closed
  boundaries -- sandbox-primitives-unavailable, an unrecognized workload,
  a missing/tampered receipt, invalid key material, and the "no key
  material in any command output" property -- using synthetic receipts and
  ``PATH`` patching so none of it needs MLX or a real sandboxed execution.

tests/test_cli.py keeps the byte-identical validate/inspect/document
behavior; this file only covers the newly-wired evaluate/promote/dispatch/
rollback/keys loop.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity, canonical_json
from auto_mlx import keys as keys_module
from auto_mlx import store_config
from auto_mlx.cli import EXIT_CONTRACT, EXIT_UNAVAILABLE, main
from auto_mlx.executor import local_sandbox_primitives_available
from auto_mlx.receipts import RawSample, Receipt


try:
    import mlx.core as _mlx_probe  # noqa: F401  -- availability probe only

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

_PRIMITIVES_AVAILABLE = local_sandbox_primitives_available()
_RUN_REAL = _MLX_AVAILABLE and _PRIMITIVES_AVAILABLE
_SKIP_REASON = "requires MLX installed and macOS local sandbox-exec primitives"


def _run_cli(*arguments: str) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = main(list(arguments))
    return status, stdout.getvalue(), stderr.getvalue()


def _write(path: Path, value: object) -> Path:
    path.write_text(canonical_json(value), encoding="utf-8")
    return path


def _toy_matmul_workload() -> FrozenWorkload:
    """The workload the CLI's built-in registry knows how to run for real.

    Matches examples/workload.json's ``toy-matmul`` contract exactly (see
    auto_mlx.runners.reference_matmul's module docstring): only this exact
    name/knobs/parameters combination resolves through the CLI's
    workload -> runner registry.
    """

    return FrozenWorkload(
        "toy-matmul",
        knobs=(
            Knob("mode", "enum", values=("eager", "compiled")),
            Knob("tile", "integer", minimum=16, maximum=32),
        ),
        parameters={"dtype": "float32", "shape": [1, 3072, 3072]},
    )


# ---------------------------------------------------------------------------
# Real-loop end-to-end proof: the actual CLI surface, the actual workload.
# ---------------------------------------------------------------------------


@unittest.skipUnless(_RUN_REAL, _SKIP_REASON)
class CLILoopEndToEndTests(unittest.TestCase):
    """Full honest chain through the real CLI: evaluate -> promote -> dispatch -> rollback."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Resolved ONCE: macOS puts TemporaryDirectory paths under /var, a
        # symlink to /private/var, and this codebase's path discipline
        # refuses to cross a symlinked ancestor (see tests/test_supervisor.py
        # SupervisorEndToEndChainTests for the identical rationale).
        self.base = Path(self.temp.name).resolve()
        self.artifact_root = self.base / "artifacts"
        self.artifact_root.mkdir()
        self.store_root = self.base / "store"
        self.key_dir = self.base / "keys"
        self.workload = _toy_matmul_workload()
        self.candidate = CandidateProposal("cli-loop-provider", self.workload, {"mode": "compiled", "tile": 24})
        self.policy = EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=60, max_output_bytes=4096)
        self.runtime = RuntimeIdentity.current()
        self.workload_path = _write(self.base / "workload.json", self.workload.to_dict())
        self.candidate_path = _write(self.base / "candidate.json", self.candidate.to_dict())
        self.policy_path = _write(self.base / "policy.json", self.policy.to_dict())
        self.runtime_path = _write(self.base / "runtime.json", self.runtime.to_dict())

    def _context_flags(self) -> list[str]:
        return [
            "--workload", str(self.workload_path),
            "--candidate", str(self.candidate_path),
            "--policy", str(self.policy_path),
            "--runtime", str(self.runtime_path),
            "--artifact-root", str(self.artifact_root),
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
        ]

    def test_full_loop_evaluate_promote_dispatch_execute_rollback(self) -> None:
        # --- evaluate: real sandboxed execution, a stored, attested receipt.
        status, stdout, stderr = _run_cli("evaluate", *self._context_flags())
        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        evaluated = json.loads(stdout)
        self.assertTrue(evaluated["ok"])
        self.assertEqual(evaluated["command"], "evaluate")
        self.assertEqual(evaluated["status"], "complete")
        self.assertTrue(evaluated["attested"], evaluated)
        self.assertEqual(evaluated["isolation_tier"], "local-sandbox-exec")
        self.assertEqual(evaluated["candidate_id"], self.candidate.candidate_id)
        self.assertEqual(evaluated["workload_hash"], self.workload.workload_hash)
        receipt_id = evaluated["receipt_id"]

        store = store_config.open_store(str(self.store_root), key_dir=str(self.key_dir))
        stored_receipt = store.get_receipt(receipt_id)
        self.assertEqual(stored_receipt.receipt_id, receipt_id)
        self.assertEqual(stored_receipt.status, "complete")
        self.assertIsNone(stored_receipt.failure)
        receipt_wire = stored_receipt.to_dict()
        gain = receipt_wire["metrics"]["gain"]
        self.assertEqual(gain, evaluated["gain"])

        # --- promote: independently re-attest and decide, matching the
        # receipt's OWN recomputed gain -- never a hardcoded outcome.
        status, stdout, stderr = _run_cli(
            "promote",
            "--receipt", receipt_id,
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
            "--artifact-root", str(self.artifact_root),
        )
        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        promoted = json.loads(stdout)
        self.assertTrue(promoted["ok"])
        self.assertTrue(promoted["attested"])
        if gain.get("improved") is True and gain.get("delta_ns", 0) > 0:
            self.assertEqual(promoted["action"], "activate")
        else:
            self.assertEqual(promoted["action"], "native_fallback")
            self.assertEqual(promoted["reason"], "gain_not_positive")
        activated = promoted["action"] == "activate"

        # --- dispatch (no --execute): reflects the just-written pointer.
        status, stdout, stderr = _run_cli("dispatch", *self._context_flags())
        self.assertEqual(status, 0, stderr)
        dispatched = json.loads(stdout)["dispatch"]
        if activated:
            self.assertEqual(dispatched["mode"], "candidate")
            self.assertEqual(dispatched["receipt_id"], receipt_id)
            self.assertEqual(dispatched["candidate_id"], self.candidate.candidate_id)
        else:
            self.assertEqual(dispatched["mode"], "native_fallback")

        # --- dispatch --execute: actually runs the selected side and
        # reports a digest matching the receipt's own recorded evidence.
        # reference_matmul is documented byte-identical across eager and
        # compiled for this workload (see its module docstring), so every
        # accepted sample -- baseline or candidate -- shares one digest.
        status, stdout, stderr = _run_cli("dispatch", *self._context_flags(), "--execute")
        self.assertEqual(status, 0, stderr)
        executed = json.loads(stdout)
        self.assertEqual(executed["execution"]["mode"], dispatched["mode"])
        expected_digest = receipt_wire["oracle"]["outcomes"][0]["actual_sha256"]
        self.assertEqual(executed["execution"]["digest"], expected_digest)
        self.assertGreater(executed["execution"]["duration_ns"], 0)

        # --- rollback: flips dispatch back to native regardless of the
        # prior decision.
        status, stdout, stderr = _run_cli("rollback", "--store", str(self.store_root), "--key-dir", str(self.key_dir))
        self.assertEqual(status, 0, stderr)
        rolled_back = json.loads(stdout)
        self.assertEqual(rolled_back["action"], "native_fallback")
        self.assertEqual(rolled_back["current_decision_id"], "native_fallback")

        status, stdout, stderr = _run_cli("dispatch", *self._context_flags())
        self.assertEqual(status, 0, stderr)
        after_rollback = json.loads(stdout)["dispatch"]
        self.assertEqual(after_rollback["mode"], "native_fallback")
        self.assertEqual(after_rollback["reason"], "native_fallback_pointer")

        # --- a second evaluate with identical inputs: content-addressed
        # stable, no duplicate-store corruption.
        status, stdout, stderr = _run_cli("evaluate", *self._context_flags())
        self.assertEqual(status, 0, stderr)
        second_evaluated = json.loads(stdout)
        self.assertEqual(second_evaluated["candidate_id"], evaluated["candidate_id"])
        self.assertEqual(second_evaluated["workload_hash"], evaluated["workload_hash"])
        second_receipt = store.get_receipt(second_evaluated["receipt_id"])
        self.assertEqual(second_receipt.receipt_id, second_evaluated["receipt_id"])
        # Both receipts remain independently retrievable: no corruption.
        self.assertEqual(store.get_receipt(receipt_id).receipt_id, receipt_id)


# ---------------------------------------------------------------------------
# Fail-closed boundaries: run on every host, no MLX or real sandbox needed.
# ---------------------------------------------------------------------------


class CLILoopFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.artifact_root = self.base / "artifacts"
        self.artifact_root.mkdir()
        self.store_root = self.base / "store"
        self.key_dir = self.base / "keys"
        self.workload = _toy_matmul_workload()
        self.candidate = CandidateProposal("fail-closed-provider", self.workload, {"mode": "compiled", "tile": 24})
        self.policy = EvaluationPolicy(warmup_runs=1, measurement_runs=2, timeout_seconds=60, max_output_bytes=4096)
        self.workload_path = _write(self.base / "workload.json", self.workload.to_dict())
        self.candidate_path = _write(self.base / "candidate.json", self.candidate.to_dict())
        self.policy_path = _write(self.base / "policy.json", self.policy.to_dict())

    def _context_flags(self) -> list[str]:
        return [
            "--workload", str(self.workload_path),
            "--candidate", str(self.candidate_path),
            "--policy", str(self.policy_path),
            "--artifact-root", str(self.artifact_root),
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
        ]

    def _empty_path(self) -> contextlib.AbstractContextManager:
        empty_path_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.rmdir(empty_path_dir))
        return patch.dict(os.environ, {"PATH": empty_path_dir})

    # -- primitives-unavailable gate: identical in kind on every platform --

    def test_evaluate_fails_closed_when_sandbox_primitives_are_unavailable(self) -> None:
        with self._empty_path():
            status, stdout, stderr = _run_cli("evaluate", *self._context_flags())
        self.assertEqual(status, EXIT_UNAVAILABLE)
        self.assertEqual(stdout, "")
        diagnostic = json.loads(stderr)
        self.assertEqual(diagnostic["error"]["code"], "unavailable")
        self.assertEqual(diagnostic["error"]["details"]["status"], "unavailable")
        self.assertEqual(diagnostic["error"]["details"]["stage"], "G1")
        self.assertEqual(diagnostic["error"]["details"]["surface"], "local_sandbox")

    def test_dispatch_execute_fails_closed_when_sandbox_primitives_are_unavailable(self) -> None:
        with self._empty_path():
            status, stdout, stderr = _run_cli("dispatch", *self._context_flags(), "--execute")
        self.assertEqual(status, EXIT_UNAVAILABLE)
        self.assertEqual(stdout, "")
        diagnostic = json.loads(stderr)
        self.assertEqual(diagnostic["error"]["code"], "unavailable")
        self.assertEqual(diagnostic["error"]["details"]["stage"], "G2")
        self.assertEqual(diagnostic["error"]["details"]["surface"], "local_sandbox")

    def test_dispatch_without_execute_never_needs_sandbox_primitives(self) -> None:
        with self._empty_path():
            status, stdout, stderr = _run_cli("dispatch", *self._context_flags())
        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        result = json.loads(stdout)
        self.assertEqual(result["dispatch"]["mode"], "native_fallback")

    # -- unknown workload name: a typed diagnostic, not a generic fallback --

    def test_unknown_workload_name_is_a_typed_diagnostic(self) -> None:
        unknown_workload = FrozenWorkload(
            "not-a-known-workload",
            knobs=(
                Knob("mode", "enum", values=("eager", "compiled")),
                Knob("tile", "integer", minimum=16, maximum=32),
            ),
            parameters={"dtype": "float32", "shape": [1, 3072, 3072]},
        )
        unknown_candidate = CandidateProposal("p", unknown_workload, {"mode": "compiled", "tile": 24})
        workload_path = _write(self.base / "unknown_workload.json", unknown_workload.to_dict())
        candidate_path = _write(self.base / "unknown_candidate.json", unknown_candidate.to_dict())
        status, stdout, stderr = _run_cli(
            "evaluate",
            "--workload", str(workload_path),
            "--candidate", str(candidate_path),
            "--policy", str(self.policy_path),
            "--artifact-root", str(self.artifact_root),
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
        )
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        diagnostic = json.loads(stderr)
        self.assertEqual(diagnostic["error"]["code"], "provider_error")
        self.assertIn("not-a-known-workload", diagnostic["error"]["message"])

    # -- a synthetic, host-independent receipt for the remaining tests --

    def _store_synthetic_receipt(self) -> tuple[Receipt, bytes, FrozenWorkload, CandidateProposal, EvaluationPolicy]:
        workload = FrozenWorkload("synthetic-cli-loop", knobs=(Knob("mode", "enum", values=("eager",)),))
        candidate = CandidateProposal("grid", workload, {"mode": "eager"})
        policy = EvaluationPolicy(warmup_runs=0, measurement_runs=2)
        runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        receipt = Receipt(
            workload,
            candidate,
            policy,
            runtime,
            (RawSample(0, 100, 120, "ok", "ok", 0), RawSample(1, 110, 130, "ok", "ok", 0)),
            created_at_ns=100,
        )
        store = store_config.open_store(str(self.store_root), key_dir=str(self.key_dir))
        store.put_receipt(receipt, require_durable=True)
        key = keys_module.generate_attestation_key()
        keys_module.store_attestation_key(key, key_dir=self.key_dir)
        return receipt, key, workload, candidate, policy

    def test_promote_with_missing_receipt_fails_closed(self) -> None:
        fake_receipt_id = "0" * 64
        status, stdout, stderr = _run_cli(
            "promote",
            "--receipt", fake_receipt_id,
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
            "--artifact-root", str(self.artifact_root),
        )
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "artifact_missing")

    def test_promote_with_tampered_receipt_fails_closed(self) -> None:
        receipt, _key, _workload, _candidate, _policy = self._store_synthetic_receipt()
        receipt_path = self.store_root / "receipts" / f"{receipt.receipt_id}.json"
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        # Keeps the ORIGINAL receipt_id, exactly as a naive post-hoc file
        # edit would: Receipt.from_dict independently recomputes the body
        # hash and refuses it before anything else is even consulted.
        payload["raw_samples"][0]["duration_ns"] = 1
        receipt_path.write_text(canonical_json(payload), encoding="utf-8")
        status, stdout, stderr = _run_cli(
            "promote",
            "--receipt", receipt.receipt_id,
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
            "--artifact-root", str(self.artifact_root),
        )
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "identity_mismatch")

    def test_promote_with_wrong_key_fails_closed(self) -> None:
        receipt, _key, _workload, _candidate, _policy = self._store_synthetic_receipt()
        key_file = self.key_dir / keys_module.KEY_FILE_NAME
        os.chmod(key_file, 0o644)
        try:
            status, stdout, stderr = _run_cli(
                "promote",
                "--receipt", receipt.receipt_id,
                "--store", str(self.store_root),
                "--key-dir", str(self.key_dir),
                "--artifact-root", str(self.artifact_root),
            )
        finally:
            os.chmod(key_file, 0o600)
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "key_material_invalid")

    def test_rollback_and_keys_ensure_never_need_mlx_or_a_prior_decision(self) -> None:
        status, stdout, stderr = _run_cli("rollback", "--store", str(self.store_root), "--key-dir", str(self.key_dir))
        self.assertEqual(status, 0, stderr)
        result = json.loads(stdout)
        self.assertEqual(result["action"], "native_fallback")
        self.assertEqual(result["current_decision_id"], "native_fallback")

        status, stdout, stderr = _run_cli("keys", "ensure", "--key-dir", str(self.key_dir))
        self.assertEqual(status, 0, stderr)
        first = json.loads(stdout)
        status, stdout, stderr = _run_cli("keys", "ensure", "--key-dir", str(self.key_dir))
        self.assertEqual(status, 0, stderr)
        second = json.loads(stdout)
        self.assertEqual(first["fingerprint_sha256_16"], second["fingerprint_sha256_16"])
        self.assertEqual(first["key_path"], second["key_path"])

    def test_no_command_output_ever_contains_key_material(self) -> None:
        receipt, key, workload, candidate, policy = self._store_synthetic_receipt()
        key_hex = key.hex()
        workload_path = _write(self.base / "synthetic_workload.json", workload.to_dict())
        candidate_path = _write(self.base / "synthetic_candidate.json", candidate.to_dict())
        policy_path = _write(self.base / "synthetic_policy.json", policy.to_dict())

        transcripts: list[tuple[int, str, str]] = [
            _run_cli("keys", "ensure", "--key-dir", str(self.key_dir)),
            _run_cli(
                "promote",
                "--receipt", receipt.receipt_id,
                "--store", str(self.store_root),
                "--key-dir", str(self.key_dir),
                "--artifact-root", str(self.artifact_root),
            ),
            _run_cli(
                "dispatch",
                "--workload", str(workload_path),
                "--candidate", str(candidate_path),
                "--policy", str(policy_path),
                "--artifact-root", str(self.artifact_root),
                "--store", str(self.store_root),
                "--key-dir", str(self.key_dir),
            ),
            _run_cli("rollback", "--store", str(self.store_root), "--key-dir", str(self.key_dir)),
        ]
        for status, stdout, stderr in transcripts:
            self.assertNotIn(key_hex, stdout)
            self.assertNotIn(key_hex, stderr)
            self.assertNotIn(key_hex.upper(), stdout)
            self.assertNotIn(key_hex.upper(), stderr)


if __name__ == "__main__":
    unittest.main()
