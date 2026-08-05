"""Tests for the supervisor/attestation boundary: auto_mlx.keys,
auto_mlx.supervisor, auto_mlx.store_config, and the resulting
Receipt.from_observation_bundle status fix in auto_mlx.receipts.
"""

from __future__ import annotations

import ast
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_mlx import (
    CandidateProposal,
    EvaluationPolicy,
    FrozenWorkload,
    Knob,
    RuntimeIdentity,
)
from auto_mlx import keys as keys_module
from auto_mlx import store_config
from auto_mlx.dispatch import CANDIDATE_MODE, DEFAULT_MAX_AGE_NS, NATIVE_MODE, dispatch
from auto_mlx.errors import ContractError, FailureCode, KeyMaterialError, StoreConfigError, SupervisorRefusalError
from auto_mlx.evaluator import Evaluator
from auto_mlx.executor import (
    ExecutionPolicy,
    ExecutionStatus,
    TrustedRunnerRegistry,
    build_execution_plan,
    local_sandbox_primitives_available,
)
from auto_mlx.oracle import ExactOutputOracle
from auto_mlx.promotion import ACTIVATE, NATIVE, activate, make_promotion_decision, rollback
from auto_mlx.receipts import (
    ContentAddressedStore,
    RawSample,
    Receipt,
    receipt_attestation,
    recompute_receipt_fields,
    validate_receipt,
)
from auto_mlx.runners import BASELINE_RUNNER_ID, CANDIDATE_RUNNER_ID, register_reference_matmul_runners
from auto_mlx.sandbox import LocalSandboxAuthority, LocalSandboxProvider
from _wave_b_fixtures import build_evaluator_bundle_receipt
from auto_mlx.supervisor import attest_receipt


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


# ---------------------------------------------------------------------------
# Key management
# ---------------------------------------------------------------------------


class KeyManagementTests(unittest.TestCase):
    def test_generate_produces_distinct_32_byte_keys(self) -> None:
        first = keys_module.generate_attestation_key()
        second = keys_module.generate_attestation_key()
        self.assertEqual(len(first), keys_module.ATTESTATION_KEY_BYTES)
        self.assertEqual(len(second), keys_module.ATTESTATION_KEY_BYTES)
        self.assertNotEqual(first, second)

    def test_resolve_key_dir_precedence_explicit_env_default(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            explicit = Path(root).resolve() / "explicit"
            env_dir = Path(root).resolve() / "env"
            with patch.dict(os.environ, {keys_module.KEY_DIR_ENV: str(env_dir)}):
                self.assertEqual(keys_module.resolve_key_dir(str(explicit)), explicit)
                self.assertEqual(keys_module.resolve_key_dir(), env_dir)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(keys_module.KEY_DIR_ENV, None)
                self.assertEqual(keys_module.resolve_key_dir(), keys_module.default_key_dir())

    def test_store_then_load_round_trips_with_strict_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            key_dir = Path(root).resolve() / "keys"
            key = keys_module.generate_attestation_key()
            path = keys_module.store_attestation_key(key, key_dir=key_dir)
            self.assertTrue(path.is_file())
            self.assertEqual(stat.S_IMODE(key_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            loaded = keys_module.load_attestation_key(key_dir=key_dir)
            self.assertEqual(loaded, key)

    def test_store_is_create_only_and_never_silently_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            key_dir = Path(root).resolve() / "keys"
            keys_module.store_attestation_key(keys_module.generate_attestation_key(), key_dir=key_dir)
            with self.assertRaises(FileExistsError):
                keys_module.store_attestation_key(keys_module.generate_attestation_key(), key_dir=key_dir)

    def test_load_rejects_missing_key_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(KeyMaterialError) as context:
                keys_module.load_attestation_key(key_dir=Path(root).resolve() / "nope")
            self.assertEqual(context.exception.code, FailureCode.KEY_MATERIAL_MISSING)

    def test_load_rejects_wrong_directory_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            key_dir = Path(root).resolve() / "keys"
            keys_module.store_attestation_key(keys_module.generate_attestation_key(), key_dir=key_dir)
            os.chmod(key_dir, 0o755)
            with self.assertRaises(KeyMaterialError) as context:
                keys_module.load_attestation_key(key_dir=key_dir)
            self.assertEqual(context.exception.code, FailureCode.KEY_MATERIAL_INVALID)

    def test_load_rejects_wrong_file_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            key_dir = Path(root).resolve() / "keys"
            path = keys_module.store_attestation_key(keys_module.generate_attestation_key(), key_dir=key_dir)
            os.chmod(path, 0o644)
            with self.assertRaises(KeyMaterialError) as context:
                keys_module.load_attestation_key(key_dir=key_dir)
            self.assertEqual(context.exception.code, FailureCode.KEY_MATERIAL_INVALID)

    def test_load_rejects_symlinked_key_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root).resolve()
            real_dir = base / "real-keys"
            keys_module.store_attestation_key(keys_module.generate_attestation_key(), key_dir=real_dir)
            link = base / "linked-keys"
            link.symlink_to(real_dir)
            with self.assertRaises(KeyMaterialError):
                keys_module.load_attestation_key(key_dir=link)

    def test_load_rejects_symlinked_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root).resolve()
            key_dir = base / "keys"
            keys_module.store_attestation_key(keys_module.generate_attestation_key(), key_dir=key_dir)
            key_file = key_dir / keys_module.KEY_FILE_NAME
            elsewhere = base / "elsewhere.key"
            elsewhere.write_bytes(keys_module.generate_attestation_key())
            os.chmod(elsewhere, 0o600)
            key_file.unlink()
            os.symlink(elsewhere, key_file)
            with self.assertRaises(KeyMaterialError):
                keys_module.load_attestation_key(key_dir=key_dir)

    def test_load_rejects_empty_or_short_key(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            key_dir = Path(root).resolve() / "keys"
            keys_module.store_attestation_key(keys_module.generate_attestation_key(), key_dir=key_dir)
            key_file = key_dir / keys_module.KEY_FILE_NAME
            key_file.write_bytes(b"short")
            os.chmod(key_file, 0o600)
            with self.assertRaises(KeyMaterialError):
                keys_module.load_attestation_key(key_dir=key_dir)

    def test_ensure_attestation_key_generates_once_and_reuses(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            key_dir = Path(root).resolve() / "keys"
            first = keys_module.ensure_attestation_key(key_dir=key_dir)
            second = keys_module.ensure_attestation_key(key_dir=key_dir)
            self.assertEqual(first, second)
            self.assertEqual(len(first), keys_module.ATTESTATION_KEY_BYTES)


# ---------------------------------------------------------------------------
# Store-root convention
# ---------------------------------------------------------------------------


class StoreConfigTests(unittest.TestCase):
    def test_resolve_store_root_precedence_explicit_env_default(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root).resolve()
            explicit = base / "explicit-store"
            env_store = base / "env-store"
            with patch.dict(os.environ, {store_config.STORE_ROOT_ENV: str(env_store)}):
                self.assertEqual(store_config.resolve_store_root(str(explicit)), explicit)
                self.assertEqual(store_config.resolve_store_root(), env_store)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(store_config.STORE_ROOT_ENV, None)
                self.assertEqual(
                    store_config.resolve_store_root(),
                    Path.cwd() / store_config.DEFAULT_STORE_DIR_NAME,
                )

    def test_nested_roots_are_rejected_either_direction(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root).resolve()
            with self.assertRaises(StoreConfigError):
                store_config.validate_disjoint_roots(base / "a", base / "a" / "b")
            with self.assertRaises(StoreConfigError):
                store_config.validate_disjoint_roots(base / "a" / "b", base / "a")
            with self.assertRaises(StoreConfigError):
                store_config.validate_disjoint_roots(base / "a", base / "a")
            store_config.validate_disjoint_roots(base / "a", base / "b")  # disjoint: does not raise

    def test_open_store_enforces_disjoint_roots_and_returns_a_content_addressed_store(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root).resolve()
            with self.assertRaises(StoreConfigError):
                store_config.open_store(base / "shared", key_dir=base / "shared" / "keys")
            store = store_config.open_store(base / "store", key_dir=base / "keys")
            self.assertIsInstance(store, ContentAddressedStore)
            self.assertEqual(Path(store.root), base / "store")


# ---------------------------------------------------------------------------
# Supervisor: independent recompute, identity, and evidence-chain checks
# ---------------------------------------------------------------------------


class SupervisorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = FrozenWorkload("supervisor-test", knobs=(Knob("mode", "enum", values=("eager",)),))
        self.candidate = CandidateProposal("grid", self.workload, {"mode": "eager"})
        self.policy = EvaluationPolicy(warmup_runs=0, measurement_runs=2)
        self.runtime = RuntimeIdentity("python", "3.11.0", "Darwin", "arm64")
        self.key = keys_module.generate_attestation_key()
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

    def test_attest_receipt_matches_the_raw_hmac_and_lets_promotion_activate(self) -> None:
        # Wave B: promotion now gates on the independently recomputed
        # statistics verdict, which only exists for evaluator-bundle-backed
        # receipts -- self.receipt (the raw-sample lane, used by the other
        # tests in this class for simpler forged-field checks) is correctly
        # never promotable under that gate.  This test specifically checks
        # that attest_receipt() + make_promotion_decision() can reach
        # ACTIVATE, so it needs a genuinely decisive receipt -- see
        # _wave_b_fixtures.
        policy = EvaluationPolicy(warmup_runs=0, measurement_runs=2, max_measurement_runs=2)
        receipt = build_evaluator_bundle_receipt(self.workload, self.candidate, policy, self.runtime)
        tag = attest_receipt(receipt, self.key)
        self.assertEqual(tag, receipt_attestation(receipt, self.key))
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = str(Path(raw_root).resolve())
            validation = validate_receipt(
                receipt, artifact_root=artifact_root, attestation=tag, attestation_key=self.key
            )
            self.assertTrue(validation.ok)
            decision = make_promotion_decision(validation, now_ns=110, attestation_key=self.key)
        self.assertEqual(decision.action, ACTIVATE)

    def test_attest_receipt_rejects_a_non_receipt_value(self) -> None:
        with self.assertRaises(SupervisorRefusalError):
            attest_receipt(self.receipt.to_dict(), self.key)  # type: ignore[arg-type]

    def test_attest_receipt_refuses_self_consistent_forged_aggregates(self) -> None:
        # A receipt whose stored `metrics` was overridden at construction
        # time to something that does NOT match what
        # recompute_receipt_fields would derive from its own raw_samples --
        # but whose receipt_id is still internally self-consistent with
        # that forged body.  Receipt.from_dict alone cannot catch this (the
        # wire is coherent); only independent recomputation can. This is
        # exactly what the supervisor must reuse from validate_receipt, not
        # reimplement.
        computed = recompute_receipt_fields(self.receipt.raw_samples, self.policy)
        forged_metrics = dict(computed["metrics"])
        forged_metrics["gain"] = {**forged_metrics["gain"], "delta_ns": 999_999, "improved": True}
        forged = Receipt(
            self.workload,
            self.candidate,
            self.policy,
            self.runtime,
            self.receipt.raw_samples,
            created_at_ns=100,
            metrics=forged_metrics,
        )
        self.assertNotEqual(forged.receipt_id, self.receipt.receipt_id)
        with self.assertRaises(SupervisorRefusalError):
            attest_receipt(forged, self.key)

    def test_attest_receipt_refuses_oracle_mismatch(self) -> None:
        mismatched = Receipt(
            self.workload,
            self.candidate,
            self.policy,
            self.runtime,
            (
                RawSample(0, 100, 120, "wrong", "ok", 0),
                RawSample(1, 110, 130, "ok", "ok", 0),
            ),
            created_at_ns=100,
        )
        with self.assertRaises(SupervisorRefusalError):
            attest_receipt(mismatched, self.key)

    def test_attest_receipt_refuses_incomplete_samples(self) -> None:
        incomplete = Receipt(
            self.workload,
            self.candidate,
            self.policy,
            self.runtime,
            (RawSample(0, 100, 120, "ok", "ok", 0),),
            created_at_ns=100,
        )
        with self.assertRaises(SupervisorRefusalError):
            attest_receipt(incomplete, self.key)

    def test_attest_receipt_refuses_failed_status_receipt(self) -> None:
        from auto_mlx.errors import Failure

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
        with self.assertRaises(SupervisorRefusalError):
            attest_receipt(failed, self.key)

    def test_promotion_refuses_receipt_tampered_after_supervisor_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = str(Path(raw_root).resolve())
            tag = attest_receipt(self.receipt, self.key)
            tampered_dict = self.receipt.to_dict()
            tampered_dict["raw_samples"][0]["duration_ns"] = 1
            # The tampered wire keeps the ORIGINAL receipt_id, exactly as a
            # naive post-hoc file edit would -- Receipt.from_dict (called
            # inside validate_receipt) independently recomputes the body
            # hash and refuses it before the HMAC is ever consulted.
            validation = validate_receipt(
                tampered_dict, artifact_root=artifact_root, attestation=tag, attestation_key=self.key
            )
            self.assertFalse(validation.ok)
            decision = make_promotion_decision(validation, now_ns=110, attestation_key=self.key)
        self.assertNotEqual(decision.action, ACTIVATE)


# ---------------------------------------------------------------------------
# Module boundary: evaluator-side code never touches key material
# ---------------------------------------------------------------------------


class EvaluatorKeyIsolationTests(unittest.TestCase):
    """The supervisor boundary: evaluator-side code must never touch key material."""

    _BANNED_MODULES = {"auto_mlx.keys", "auto_mlx.supervisor", "keys", "supervisor"}
    _BANNED_IDENTIFIERS = {
        "attestation_key",
        "load_attestation_key",
        "generate_attestation_key",
        "store_attestation_key",
        "ensure_attestation_key",
        "receipt_attestation",
        "attest_receipt",
    }

    def _assert_source_never_references_key_material(self, relative_path: str) -> None:
        source = (SRC_ROOT / "auto_mlx" / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name, self._BANNED_MODULES, f"{relative_path} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                self.assertNotIn(module, self._BANNED_MODULES, f"{relative_path} imports from {module}")
                if node.level > 0 and module in {"keys", "supervisor"}:
                    self.fail(f"{relative_path} must not import .{module}")
        found_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        found_attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        overlap = (found_names | found_attrs) & self._BANNED_IDENTIFIERS
        self.assertFalse(overlap, f"{relative_path} references key-material identifiers: {overlap}")

    def test_evaluator_module_source_never_references_key_material(self) -> None:
        self._assert_source_never_references_key_material("evaluator.py")

    def test_executor_module_source_never_references_key_material(self) -> None:
        self._assert_source_never_references_key_material("executor.py")

    def test_sandbox_module_source_never_references_key_material(self) -> None:
        self._assert_source_never_references_key_material("sandbox.py")

    def test_importing_evaluator_module_never_pulls_in_key_or_supervisor_modules(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import auto_mlx.evaluator\n"
                "print('auto_mlx.keys' in sys.modules)\n"
                "print('auto_mlx.supervisor' in sys.modules)\n",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.splitlines(), ["False", "False"])

    def test_importing_the_top_level_package_never_pulls_in_key_or_supervisor_modules(self) -> None:
        # auto_mlx/__init__.py deliberately does NOT import .keys/.supervisor
        # at package top level: Python always runs the package __init__ before
        # any submodule import completes, so if __init__.py imported them
        # eagerly, `import auto_mlx.evaluator` (and every other submodule
        # import) would transitively load auto_mlx.keys regardless of
        # evaluator.py's own source -- silently defeating the module
        # boundary the previous two tests check for. This guards that
        # property directly against a future regression in __init__.py.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_ROOT)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import auto_mlx\n"
                "print('auto_mlx.keys' in sys.modules)\n"
                "print('auto_mlx.supervisor' in sys.modules)\n"
                "print('auto_mlx.store_config' in sys.modules)\n",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.splitlines(), ["False", "False", "False"])

    @unittest.skipUnless(_RUN_REAL, _SKIP_REASON)
    def test_real_evaluate_call_never_imports_key_or_supervisor_modules(self) -> None:
        script = (
            "import sys, tempfile\n"
            f"sys.path.insert(0, {str(SRC_ROOT)!r})\n"
            "from auto_mlx.evaluator import Evaluator\n"
            "from auto_mlx.executor import build_execution_plan, ExecutionPolicy, TrustedRunnerRegistry\n"
            "from auto_mlx.oracle import ExactOutputOracle\n"
            "from auto_mlx.runners import BASELINE_RUNNER_ID, CANDIDATE_RUNNER_ID, register_reference_matmul_runners\n"
            "from auto_mlx.sandbox import LocalSandboxAuthority, LocalSandboxProvider\n"
            "from auto_mlx import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob\n"
            "registry = TrustedRunnerRegistry()\n"
            "register_reference_matmul_runners(registry)\n"
            "workload = FrozenWorkload(\n"
            "    'key-isolation',\n"
            "    knobs=(Knob('mode', 'enum', values=('eager', 'compiled')), Knob('tile', 'integer', minimum=16, maximum=32)),\n"
            "    parameters={'dtype': 'float32', 'shape': [1, 3072, 3072]},\n"
            ")\n"
            "proposal = CandidateProposal('t', workload, {'mode': 'eager', 'tile': 16})\n"
            "with tempfile.TemporaryDirectory() as root:\n"
            "    plan = build_execution_plan(proposal, registry, BASELINE_RUNNER_ID, root)\n"
            "    record = plan.execute(ExecutionPolicy(timeout_seconds=60, max_output_bytes=4096), registry=registry, provider=LocalSandboxProvider(), authority=LocalSandboxAuthority())\n"
            "    oracle = ExactOutputOracle(record.stdout)\n"
            "    evaluator = Evaluator(\n"
            "        registry, baseline_runner_id=BASELINE_RUNNER_ID, candidate_runner_id=CANDIDATE_RUNNER_ID,\n"
            "        oracle=oracle, artifact_root=root,\n"
            # Bounded k_repetitions/max_measurement_runs: production
            # defaults (k_repetitions=50, max_measurement_runs=20) can make
            # a real inconclusive-verdict run exceed this test's subprocess
            # timeout below; this test only checks module import hygiene.
            "        policy=EvaluationPolicy(\n"
            "            warmup_runs=1, measurement_runs=1, max_measurement_runs=2, k_repetitions=5,\n"
            "            bootstrap_resamples=500, timeout_seconds=60, max_output_bytes=4096,\n"
            "        ),\n"
            "        provider=LocalSandboxProvider(), authority=LocalSandboxAuthority(),\n"
            "    )\n"
            "    evaluator.evaluate(proposal)\n"
            "print('auto_mlx.keys' in sys.modules)\n"
            "print('auto_mlx.supervisor' in sys.modules)\n"
        )
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.splitlines()[-2:], ["False", "False"])


# ---------------------------------------------------------------------------
# End-to-end library proof: real evaluation through to dispatch and rollback
# ---------------------------------------------------------------------------


@unittest.skipUnless(_RUN_REAL, _SKIP_REASON)
class SupervisorEndToEndChainTests(unittest.TestCase):
    """Full honest chain: Evaluator -> receipt -> supervisor -> promotion -> dispatch -> rollback."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        # Resolved ONCE: macOS puts TemporaryDirectory paths under /var, a
        # symlink to /private/var, and the descriptor-relative no-follow
        # walks in auto_mlx.paths / auto_mlx.keys correctly refuse to cross
        # it. Every root derived below must come from this resolved base.
        self.base = Path(self.temp.name).resolve()
        self.artifact_root = self.base / "artifacts"
        self.artifact_root.mkdir()
        self.store_root = self.base / "store"
        self.key_dir = self.base / "keys"
        self.registry = TrustedRunnerRegistry()
        register_reference_matmul_runners(self.registry)
        self.workload = FrozenWorkload(
            "toy-matmul-supervisor-e2e",
            knobs=(
                Knob("mode", "enum", values=("eager", "compiled")),
                Knob("tile", "integer", minimum=16, maximum=32),
            ),
            parameters={"dtype": "float32", "shape": [1, 3072, 3072]},
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_full_honest_chain_from_evaluation_to_dispatch_and_rollback(self) -> None:
        proposal = CandidateProposal("e2e-provider", self.workload, {"mode": "compiled", "tile": 24})

        # The oracle's expected bytes come from one real baseline execution,
        # not a hardcoded literal (matches ReferenceMatmulEvaluatorLoopTests).
        probe_plan = build_execution_plan(proposal, self.registry, BASELINE_RUNNER_ID, str(self.artifact_root))
        probe_record = probe_plan.execute(
            ExecutionPolicy(timeout_seconds=60, max_output_bytes=4096),
            registry=self.registry,
            provider=LocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        self.assertIs(probe_record.status, ExecutionStatus.SUCCESS, probe_record.failure)
        oracle = ExactOutputOracle(probe_record.stdout)

        # Bounded k_repetitions/max_measurement_runs/bootstrap_resamples --
        # see ReferenceMatmulEvaluatorLoopTests's identical comment.
        policy = EvaluationPolicy(
            warmup_runs=1, measurement_runs=2, max_measurement_runs=4, k_repetitions=5, bootstrap_resamples=500,
            timeout_seconds=60, max_output_bytes=4096,
        )
        evaluator = Evaluator(
            self.registry,
            baseline_runner_id=BASELINE_RUNNER_ID,
            candidate_runner_id=CANDIDATE_RUNNER_ID,
            oracle=oracle,
            artifact_root=str(self.artifact_root),
            policy=policy,
            provider=LocalSandboxProvider(),
            authority=LocalSandboxAuthority(),
        )
        bundle = evaluator.evaluate(proposal)
        self.assertTrue(bundle.accepted, bundle.measurements.rejection_reasons)
        self.assertTrue(bundle.promotion_eligible)

        receipt = Receipt.from_observation_bundle(
            bundle, self.workload, proposal, policy, oracle=oracle, created_at_ns=1
        )
        self.assertEqual(receipt.status, "complete")
        self.assertIsNone(receipt.failure)

        # STORE CONVENTION: resolve store/key roots through the same
        # explicit-arg/env/default precedence a future CLI wave will
        # expose, proving the roots are validated disjoint.
        with patch.dict(
            os.environ,
            {store_config.STORE_ROOT_ENV: str(self.store_root), keys_module.KEY_DIR_ENV: str(self.key_dir)},
        ):
            store = store_config.open_store()
            self.assertEqual(Path(store.root), self.store_root)
            store.put_receipt(receipt)
            key = keys_module.ensure_attestation_key()

        # SUPERVISOR: the only code path that mints the attestation.
        attestation = attest_receipt(receipt, key, artifact_root=str(self.artifact_root))
        self.assertEqual(attestation, receipt_attestation(receipt, key))

        validation = validate_receipt(
            receipt, artifact_root=str(self.artifact_root), attestation=attestation, attestation_key=key
        )
        self.assertTrue(validation.ok)

        decision = make_promotion_decision(validation, now_ns=100, attestation_key=key)
        # The decision must reflect whatever the real, evidence-based Wave B
        # statistics verdict says -- not a wished-for outcome. Assert the
        # policy logic, not a specific verdict: a decisive "improved"
        # activates; "regressed"/"inconclusive" must fall back to native
        # with the matching reason code (see auto_mlx.promotion).
        statistics = validation.recomputed["statistics"]
        self.assertIsNotNone(statistics)
        if statistics["verdict"] == "improved":
            self.assertEqual(decision.action, ACTIVATE)
        else:
            self.assertIn(statistics["verdict"], {"regressed", "inconclusive"})
            self.assertEqual(decision.action, NATIVE)
            self.assertEqual(decision.reason, statistics["verdict"])

        activated = activate(store, validation, artifact_root=str(self.artifact_root), attestation_key=key, now_ns=100)
        self.assertEqual(activated.action, decision.action)

        runtime = RuntimeIdentity.current()
        result = dispatch(
            store,
            self.workload,
            proposal,
            policy,
            runtime,
            artifact_root=str(self.artifact_root),
            attestation_key=key,
            now_ns=110,
            max_age_ns=DEFAULT_MAX_AGE_NS,
        )
        if activated.action == ACTIVATE:
            self.assertEqual(result.mode, CANDIDATE_MODE)
            self.assertEqual(result.candidate_id, proposal.candidate_id)
            self.assertEqual(result.receipt_id, receipt.receipt_id)
        else:
            self.assertEqual(result.mode, NATIVE_MODE)

        # ROLLBACK: flips dispatch back to native regardless of the prior
        # decision, and the prior receipt/decision remain intact evidence.
        rolled_back = rollback(store, now_ns=120)
        self.assertEqual(rolled_back.action, NATIVE)
        self.assertEqual(store.current_decision_id(), "native_fallback")
        self.assertEqual(store.get_receipt(receipt.receipt_id), receipt)

        after_rollback = dispatch(
            store,
            self.workload,
            proposal,
            policy,
            runtime,
            artifact_root=str(self.artifact_root),
            attestation_key=key,
            now_ns=130,
            max_age_ns=DEFAULT_MAX_AGE_NS,
        )
        self.assertEqual(after_rollback.mode, NATIVE_MODE)
        self.assertEqual(after_rollback.reason, "native_fallback_pointer")

    def test_wrong_permission_key_file_makes_load_refuse(self) -> None:
        key = keys_module.generate_attestation_key()
        path = keys_module.store_attestation_key(key, key_dir=self.key_dir)
        os.chmod(path, 0o644)
        with self.assertRaises(KeyMaterialError) as context:
            keys_module.load_attestation_key(key_dir=self.key_dir)
        self.assertEqual(context.exception.code, FailureCode.KEY_MATERIAL_INVALID)


if __name__ == "__main__":
    unittest.main()
