"""Wave B integration tests: real subprocess evaluation, sequential sampling,
calibration, and supervisor recompute-mismatch refusal.

Two families:

- ``SequentialSamplingTests`` / ``EndToEndReferenceMatmulTests`` run real
  sandboxed subprocess executions (``skipUnless`` the local sandbox
  primitives / MLX are available) -- no mocks for the evaluation loop
  itself.
- ``CalibrationTests`` / ``SupervisorRecomputeMismatchTests`` exercise the
  promotion/validation layer directly against hand-built (but genuinely
  evidentiary) evaluator-bundle receipts, mirroring
  ``tests/_wave_b_fixtures.py``.
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

from auto_mlx import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity
from auto_mlx.evaluator import Evaluator
from auto_mlx.executor import (
    ExecutionStatus,
    IsolatedProcess,
    IsolationAuthority,
    IsolationClaim,
    IsolationProvider,
    TrustedRunner,
    TrustedRunnerRegistry,
    build_execution_plan,
    local_sandbox_primitives_available,
)
from auto_mlx.oracle import ExactOutputOracle
from auto_mlx.promotion import make_promotion_decision
from auto_mlx.receipts import ContentAddressedStore, Receipt, receipt_attestation, validate_receipt
from auto_mlx.runners import BASELINE_RUNNER_ID, CANDIDATE_RUNNER_ID, register_reference_matmul_runners
from auto_mlx.sandbox import LocalSandboxAuthority, LocalSandboxProvider
from auto_mlx.statistics import VERDICT_IMPROVED, VERDICT_INCONCLUSIVE

from _wave_b_fixtures import build_evaluator_bundle_receipt


_PRIMITIVES_AVAILABLE = local_sandbox_primitives_available()
_SKIP_REASON = "requires macOS local sandbox-exec primitives"

try:
    import mlx.core as _mlx_probe  # noqa: F401

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False


class _FixtureIsolationProvider(IsolationProvider):
    def __init__(self) -> None:
        super().__init__("wave-b-integration-isolation", "3" * 64, supports_evaluator_owned_launch=True)

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
        return IsolatedProcess(process, self._claim("4" * 64))


class _FixtureIsolationAuthority(IsolationAuthority):
    def __init__(self) -> None:
        super().__init__("wave-b-integration-verifier", "5" * 64, production_eligible=False)

    def verify(self, provider, process, claim: IsolationClaim):
        return self._attest(provider, claim)


def _nominal_preflight() -> dict:
    reading = {
        "state": "nominal",
        "cpu_speed_limit_percent": 100,
        "cpu_scheduler_limit_percent": None,
        "thermal_pressure_level": None,
        "detail": "",
    }
    return {"initial": reading, "final": reading, "retried": False, "thermally_suspect": False}


_FIXED_TIMING_RUNNER = """
import json
import os
import sys

report_ns = None
for arg in sys.argv[1:]:
    if arg.startswith("--report-ns="):
        report_ns = int(arg.split("=", 1)[1])
if report_ns is None:
    raise SystemExit("missing --report-ns")

config_path = os.environ.get("AUTO_MLX_CONFIG_PATH")
with open(config_path, "rb") as handle:
    raw = handle.read()
config = json.loads(raw)
if set(config) != {"mode"} or config["mode"] not in ("safe",):
    raise SystemExit("unexpected config")

# Always report exactly ONE iteration, regardless of the requested
# AUTO_MLX_K_REPETITIONS: this keeps the reported sum tiny and safely
# below any real runner_elapsed_ns (Python interpreter startup alone is
# normally several milliseconds), so compute_sample_timing's
# forged-timing cross-check always trusts it -- reporting K copies of
# report_ns would multiply the sum past the real wall-clock span and
# force an (unwanted, for this fixture) forged_timing fallback to noisy
# real subprocess timing.
print("auto_mlx_runner_warmup_complete", file=sys.stderr)
payload = json.dumps({"k": 1, "iterations_ns": [report_ns]})
print(f"auto_mlx_runner_iter_timings_v1 {payload}", file=sys.stderr, flush=True)
print("ok")
"""


@unittest.skipUnless(_PRIMITIVES_AVAILABLE, _SKIP_REASON)
class SequentialSamplingTests(unittest.TestCase):
    """Real subprocess evaluations with fully controlled, zero-variance timing.

    A fixed ``--report-ns=`` argument (part of each TrustedRunner's pinned,
    verified argv -- not caller-controlled config) makes every sample's
    reported iteration timing identical and deterministic, so the resulting
    per-block differences have zero variance and the BCa bootstrap CI
    collapses to an exact point every time -- eliminating flakiness from
    real scheduling/thermal noise while still exercising a real sandboxed
    subprocess per sample.
    """

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "fixed_timing_runner.py"
        self.script.write_text(_FIXED_TIMING_RUNNER, encoding="utf-8")
        self.workload = FrozenWorkload("wave-b-sequential", knobs=(Knob("mode", "enum", values=("safe",)),))
        self.proposal = CandidateProposal("fixture-provider", self.workload, {"mode": "safe"})
        self.python_artifact = str(Path(sys.executable).resolve())

    def _registry(self, *, baseline_ns: int, candidate_ns: int) -> TrustedRunnerRegistry:
        registry = TrustedRunnerRegistry()
        registry.register_command(
            "baseline",
            (sys.executable, str(self.script), f"--report-ns={baseline_ns}"),
            artifact_paths=(str(self.script), self.python_artifact),
        )
        registry.register_command(
            "candidate",
            (sys.executable, str(self.script), f"--report-ns={candidate_ns}"),
            artifact_paths=(str(self.script), self.python_artifact),
        )
        return registry

    def _evaluate(self, *, baseline_ns: int, candidate_ns: int, policy: EvaluationPolicy, oracle: ExactOutputOracle | None = None):
        registry = self._registry(baseline_ns=baseline_ns, candidate_ns=candidate_ns)
        evaluator = Evaluator(
            registry,
            baseline_runner_id="baseline",
            candidate_runner_id="candidate",
            oracle=oracle or ExactOutputOracle(b"ok\n"),
            artifact_root=str(self.root),
            policy=policy,
            thermal_preflight=_nominal_preflight,
            provider=_FixtureIsolationProvider(),
            authority=_FixtureIsolationAuthority(),
        )
        return evaluator.evaluate(self.proposal)

    def test_decisive_verdict_stops_at_the_starting_block_count(self) -> None:
        # warmup_runs must be >= 1: bundle-level isolation identity is
        # derived exclusively from the first warmup's real VerifiedIsolation
        # evidence (see Evaluator.evaluate); zero warmups leaves it fully
        # unbound and every measurement sample rejected regardless of
        # timing.
        policy = EvaluationPolicy(
            warmup_runs=1, measurement_runs=2, max_measurement_runs=10, timeout_seconds=30, max_output_bytes=4096,
        )
        bundle = self._evaluate(baseline_ns=2_000_000, candidate_ns=1_000_000, policy=policy)
        self.assertTrue(bundle.accepted, bundle.measurements.rejection_reasons)
        self.assertIsNotNone(bundle.statistics)
        self.assertEqual(bundle.statistics["verdict"], VERDICT_IMPROVED)
        # No extension: block count stayed at the policy's starting count.
        self.assertEqual(bundle.measurement_block_count, policy.measurement_runs)
        self.assertEqual(bundle.statistics["block_count_used"], policy.measurement_runs)
        self.assertEqual(len(bundle.thermal_blocks), policy.measurement_runs)

    def test_inconclusive_verdict_extends_to_the_cap(self) -> None:
        policy = EvaluationPolicy(
            warmup_runs=1, measurement_runs=2, max_measurement_runs=4, timeout_seconds=30, max_output_bytes=4096,
        )
        # Identical reported timings on both arms: the true difference is
        # exactly zero every block, so the verdict can never become
        # decisive -- sequential sampling must extend all the way to the
        # policy's cap and stop there.
        bundle = self._evaluate(baseline_ns=1_000_000, candidate_ns=1_000_000, policy=policy)
        self.assertTrue(bundle.accepted, bundle.measurements.rejection_reasons)
        self.assertIsNotNone(bundle.statistics)
        self.assertEqual(bundle.statistics["verdict"], VERDICT_INCONCLUSIVE)
        self.assertEqual(bundle.measurement_block_count, policy.max_measurement_runs)
        self.assertEqual(bundle.statistics["block_count_used"], policy.max_measurement_runs)
        self.assertEqual(len(bundle.thermal_blocks), policy.max_measurement_runs)
        self.assertEqual(bundle.statistics["ci_lower_ns"], 0)
        self.assertEqual(bundle.statistics["ci_upper_ns"], 0)

    def test_receipt_from_sequential_bundle_is_promotable(self) -> None:
        policy = EvaluationPolicy(
            warmup_runs=1, measurement_runs=2, max_measurement_runs=10, timeout_seconds=30, max_output_bytes=4096,
        )
        # Reuse the exact same oracle instance the Evaluator was
        # constructed with: _observation_bundle_to_wire checks the bundle
        # retains the constructor's original oracle by identity, not
        # equal-content reconstruction.
        oracle = ExactOutputOracle(b"ok\n")
        bundle = self._evaluate(baseline_ns=2_000_000, candidate_ns=1_000_000, policy=policy, oracle=oracle)
        receipt = Receipt.from_observation_bundle(
            bundle, self.workload, self.proposal, policy, oracle=oracle, created_at_ns=100,
        )
        self.assertEqual(receipt.status, "complete")
        self.assertEqual(receipt.statistics["verdict"], VERDICT_IMPROVED)
        key = b"wave-b-sequential-key"
        tag = receipt_attestation(receipt, key)
        with tempfile.TemporaryDirectory() as raw_root:
            # Resolved: macOS TemporaryDirectory paths live under /var, a
            # symlink to /private/var, and validate_receipt's descriptor-
            # relative no-follow artifact_root walk correctly refuses to
            # cross it -- see tests/test_promotion.py's identical note.
            artifact_root = str(Path(raw_root).resolve())
            validation = validate_receipt(receipt, artifact_root=artifact_root, attestation=tag, attestation_key=key)
            self.assertTrue(validation.ok)
            decision = make_promotion_decision(validation, now_ns=200, attestation_key=key)
        self.assertEqual(decision.action, "activate")


class CalibrationTests(unittest.TestCase):
    """A/A calibration receipts (policy.calibration=True) are never promotable."""

    def setUp(self) -> None:
        self.workload = FrozenWorkload("wave-b-calibration", knobs=(Knob("mode", "enum", values=("eager",)),))
        self.candidate = CandidateProposal("grid", self.workload, {"mode": "eager"})
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.key = b"wave-b-calibration-key"

    def test_calibration_receipt_reports_the_noise_floor_but_cannot_activate(self) -> None:
        # A true A/A run has candidate == baseline: zero true difference,
        # but the fixture's zero-variance construction would trivially
        # report a zero-width, zero-centered CI, which is not an
        # interesting noise-floor signal. Instead this directly asserts the
        # POLICY gate (calibration=True is unpromotable regardless of
        # verdict) -- the real noise-floor measurement is exercised by the
        # actual `auto-mlx evaluate --calibrate` CLI path (see the manual
        # validation run in the task return notes), which uses real,
        # independently-varying subprocess timings.
        policy = EvaluationPolicy(
            warmup_runs=0, measurement_runs=2, max_measurement_runs=2, calibration=True,
        )
        # Even a receipt whose CI would otherwise read as decisively
        # "improved" must never activate once policy.calibration is True.
        receipt = build_evaluator_bundle_receipt(self.workload, self.candidate, policy, self.runtime)
        self.assertTrue(receipt.statistics["calibration"])
        self.assertEqual(receipt.statistics["verdict"], VERDICT_IMPROVED)
        tag = receipt_attestation(receipt, self.key)
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = str(Path(raw_root).resolve())
            validation = validate_receipt(receipt, artifact_root=artifact_root, attestation=tag, attestation_key=self.key)
            self.assertTrue(validation.ok)
            decision = make_promotion_decision(validation, now_ns=100, attestation_key=self.key)
        self.assertEqual(decision.action, "native_fallback")
        self.assertEqual(decision.reason, "calibration_receipt_not_promotable")


class SupervisorRecomputeMismatchTests(unittest.TestCase):
    """Tampered statistics must fail independent recomputation, not be trusted.

    A receipt's evaluator_bundle.statistics is recomputed from raw
    per-sample evidence and cross-checked at EVERY parse
    (``Receipt.__init__`` always calls ``recompute_receipt_fields``,
    whether constructing fresh or via ``Receipt.from_dict``) -- so a
    tampered statistics field can never even be loaded back into a
    ``Receipt`` object, let alone attested or promoted.  These tests
    exercise that boundary at the two places tampering actually enters a
    real deployment: raw-document validation
    (``auto_mlx.receipts.validate_receipt``, which safely wraps a failed
    reconstruction into an invalid ``ReceiptValidation`` rather than
    raising) and on-disk store reads (``ContentAddressedStore.get_receipt``).
    """

    def setUp(self) -> None:
        self.workload = FrozenWorkload("wave-b-mismatch", knobs=(Knob("mode", "enum", values=("eager",)),))
        self.candidate = CandidateProposal("grid", self.workload, {"mode": "eager"})
        self.policy = EvaluationPolicy(warmup_runs=0, measurement_runs=2, max_measurement_runs=2)
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.key = b"wave-b-mismatch-key"
        self.receipt = build_evaluator_bundle_receipt(self.workload, self.candidate, self.policy, self.runtime)

    def test_tampered_verdict_field_fails_independent_recomputation(self) -> None:
        data = self.receipt.to_dict()
        self.assertEqual(data["statistics"]["verdict"], VERDICT_IMPROVED)
        data["evaluator_bundle"]["statistics"]["verdict"] = "regressed"
        data["statistics"]["verdict"] = "regressed"
        # validate_receipt never raises on a malformed/inconsistent
        # document -- Receipt.from_dict's own recompute-and-compare fails
        # inside the try/except and comes back as an invalid validation,
        # never a trusted (or even successfully constructed) Receipt.
        result = validate_receipt(data)
        self.assertFalse(result.valid)
        self.assertIsNone(result.receipt)

    def test_tampered_ci_bound_is_caught_even_when_self_consistent(self) -> None:
        data = self.receipt.to_dict()
        # Tamper the CI bound consistently at both the top-level summary
        # and its evaluator_bundle source -- an attacker controlling the
        # whole document, not just one copy of the number.  Only a
        # from-scratch bootstrap recomputation over the untouched raw
        # per-sample iteration timings can catch this.
        data["evaluator_bundle"]["statistics"]["ci_lower_ns"] = 1
        data["statistics"]["ci_lower_ns"] = 1
        result = validate_receipt(data)
        self.assertFalse(result.valid)

    def test_stored_receipt_with_tampered_statistics_cannot_be_read_back(self) -> None:
        from auto_mlx.canonical import canonical_json
        from auto_mlx.errors import ContractError

        with tempfile.TemporaryDirectory() as raw_root:
            store = ContentAddressedStore(raw_root)
            store.put_receipt(self.receipt)
            receipt_path = Path(store.root) / "receipts" / f"{self.receipt.receipt_id}.json"
            tampered = self.receipt.to_dict()
            tampered["evaluator_bundle"]["statistics"]["bootstrap_resamples"] += 1
            tampered["statistics"]["bootstrap_resamples"] += 1
            # Canonically encoded (unlike a plain json.dumps tamper) so this
            # specifically exercises the statistics recompute-and-compare,
            # not just the separate canonical-bytes-on-disk check.
            receipt_path.write_text(canonical_json(tampered), encoding="utf-8")
            with self.assertRaises(ContractError):
                store.get_receipt(self.receipt.receipt_id)

    def test_supervisor_never_attests_a_receipt_object_it_did_not_itself_validate(self) -> None:
        """attest_receipt only ever accepts a real Receipt, never a raw/tampered mapping."""

        from auto_mlx.supervisor import attest_receipt
        from auto_mlx.errors import SupervisorRefusalError

        data = self.receipt.to_dict()
        data["evaluator_bundle"]["statistics"]["confidence_bps"] = 9999
        with self.assertRaises(SupervisorRefusalError):
            attest_receipt(data, self.key)  # type: ignore[arg-type]


@unittest.skipUnless(_PRIMITIVES_AVAILABLE and _MLX_AVAILABLE, "requires MLX and macOS local sandbox-exec primitives")
class EndToEndReferenceMatmulTests(unittest.TestCase):
    """Real end-to-end: the actual reference_matmul runner under Evaluator.evaluate()."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = TrustedRunnerRegistry()
        register_reference_matmul_runners(self.registry)
        self.workload = FrozenWorkload(
            "toy-matmul-wave-b-e2e",
            knobs=(Knob("mode", "enum", values=("eager", "compiled")), Knob("tile", "integer", minimum=16, maximum=32)),
            parameters={"dtype": "float32", "shape": [1, 3072, 3072]},
        )

    def test_real_evaluate_populates_a_statistics_verdict(self) -> None:
        proposal = CandidateProposal("test-provider", self.workload, {"mode": "compiled", "tile": 24})
        probe_plan = build_execution_plan(proposal, self.registry, BASELINE_RUNNER_ID, str(self.root))
        from auto_mlx.executor import ExecutionPolicy

        probe_record = probe_plan.execute(
            ExecutionPolicy(timeout_seconds=60, max_output_bytes=4096),
            registry=self.registry,
            provider=LocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        self.assertIs(probe_record.status, ExecutionStatus.SUCCESS, probe_record.failure)
        oracle = ExactOutputOracle(probe_record.stdout)

        # Small, bounded policy: real timing noise on this host decides the
        # verdict (this is not a performance claim -- see docs/measurement.md),
        # but the cap keeps this test's wall-clock bounded regardless.
        policy = EvaluationPolicy(
            warmup_runs=1, measurement_runs=2, max_measurement_runs=4, k_repetitions=5,
            bootstrap_resamples=500, timeout_seconds=60, max_output_bytes=4096,
        )
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
        self.assertTrue(bundle.accepted, bundle.measurements.rejection_reasons)
        self.assertIsNotNone(bundle.statistics)
        self.assertIn(bundle.statistics["verdict"], {"improved", "regressed", "inconclusive"})
        self.assertGreaterEqual(bundle.statistics["block_count_used"], policy.measurement_runs)
        self.assertLessEqual(bundle.statistics["block_count_used"], policy.max_measurement_runs)

        receipt = Receipt.from_observation_bundle(
            bundle, self.workload, proposal, policy, oracle=oracle, created_at_ns=100,
        )
        self.assertEqual(receipt.status, "complete")
        self.assertEqual(receipt.statistics, bundle.statistics)
        validation = validate_receipt(receipt)
        self.assertTrue(validation.valid, [f.message for f in validation.failures])
        self.assertEqual(validation.recomputed["statistics"], bundle.statistics)


if __name__ == "__main__":
    unittest.main()
