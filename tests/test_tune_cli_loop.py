"""Tests for the real local `auto-mlx tune`/`history` CLI loop.

Two groups, matching tests/test_cli_loop.py's convention:

- ``TuneCLIEndToEndTests`` (``skipUnless`` MLX + local sandbox primitives):
  a real, sandboxed `tune` run over a small slice of the actual toy-matmul
  knob grid, followed by `history` against the same store.
- ``TuneCLIFailClosedTests`` (no skip: runs on every host): the sandbox-
  primitives-unavailable gate, an unrecognized workload, and the "history
  never needs the sandbox" property.
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

from auto_mlx import CandidateProposal, FrozenWorkload, Knob, RuntimeIdentity, canonical_json
from auto_mlx.cli import EXIT_CONTRACT, EXIT_UNAVAILABLE, main
from auto_mlx.executor import local_sandbox_primitives_available
from auto_mlx.providers import DeclarativeProvider
from auto_mlx.tune import ENTRANT_STATUSES, STATUS_UNRESOLVED_BUDGET


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
    return FrozenWorkload(
        "toy-matmul",
        knobs=(
            Knob("mode", "enum", values=("eager", "compiled")),
            Knob("tile", "integer", minimum=16, maximum=32),
        ),
        parameters={"dtype": "float32", "shape": [1, 3072, 3072]},
    )


@unittest.skipUnless(_RUN_REAL, _SKIP_REASON)
class TuneCLIEndToEndTests(unittest.TestCase):
    """A real, sandboxed `tune` run over a small slice of the real knob grid."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Resolved ONCE: see tests/test_cli_loop.py's identical rationale
        # (macOS TemporaryDirectory paths live under a symlinked /var).
        self.base = Path(self.temp.name).resolve()
        self.artifact_root = self.base / "artifacts"
        self.artifact_root.mkdir()
        self.store_root = self.base / "store"
        self.key_dir = self.base / "keys"
        self.workload = _toy_matmul_workload()
        # Deliberately small: --max-candidates keeps this to 2 of the 34
        # legal (mode, tile) configs, and a small measurement_runs/
        # max_measurement_runs/k_repetitions/bootstrap_resamples bounds the
        # racing ladder's redo cost so this finishes in well under a minute
        # even though it runs real sandboxed MLX subprocesses.
        self.provider = DeclarativeProvider(
            "tune-cli-loop-provider",
            ({"mode": "eager", "tile": 16}, {"mode": "compiled", "tile": 24}),
        )
        self.workload_path = _write(self.base / "workload.json", self.workload.to_dict())
        self.provider_path = _write(self.base / "provider.json", self.provider.to_dict())

    def _context_flags(self) -> list[str]:
        return [
            "--workload", str(self.workload_path),
            "--provider", str(self.provider_path),
            "--artifact-root", str(self.artifact_root),
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
            "--samples", "2",
            "--max-measurement-runs", "3",
            "--k-repetitions", "3",
            "--bootstrap-resamples", "200",
            "--timeout-seconds", "60",
            "--max-output-bytes", "4096",
        ]

    def test_real_tune_then_history(self) -> None:
        status, stdout, stderr = _run_cli("tune", *self._context_flags())
        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        summary = json.loads(stdout)
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["command"], "tune")
        self.assertEqual(summary["workload_hash"], self.workload.workload_hash)
        self.assertEqual(summary["provider_id"], self.provider.provider_id)

        entrants = summary["entrants"]
        self.assertEqual(len(entrants), 2)
        self.assertEqual(summary["prefilter"]["considered"], 2)
        self.assertEqual(summary["prefilter"]["pruned"], [])
        self.assertEqual(summary["prefilter"]["raced_count"], 2)
        self.assertEqual(summary["baseline"]["status"], "floor")

        for entrant in entrants:
            self.assertIn(entrant["status"], ENTRANT_STATUSES)
            # Every measured candidate gets a full attested receipt through
            # the existing pipeline -- unless the race never got to spend
            # any budget on it at all (not exercised here: no budget cap is
            # set, so every entrant is genuinely measured).
            self.assertNotEqual(entrant["status"], STATUS_UNRESOLVED_BUDGET)
            self.assertIsNotNone(entrant["receipt_id"])
            self.assertGreaterEqual(entrant["block_count_used"], 2)
            self.assertLessEqual(entrant["block_count_used"], 3)
            self.assertIsNotNone(entrant["statistics"])

        self.assertGreater(summary["budget"]["blocks_spent"], 0)
        self.assertFalse(summary["budget"]["exhausted"])

        # Every entrant's receipt is independently retrievable and complete.
        from auto_mlx import store_config

        store = store_config.open_store(str(self.store_root), key_dir=str(self.key_dir))
        for entrant in entrants:
            receipt = store.get_receipt(entrant["receipt_id"])
            self.assertEqual(receipt.receipt_id, entrant["receipt_id"])
            self.assertEqual(receipt.status, "complete")

        # --- history: the just-stored summary is listed back for this
        # exact (workload, runtime) pair.
        status, stdout, stderr = _run_cli(
            "history",
            "--workload", str(self.workload_path),
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
        )
        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        history = json.loads(stdout)
        self.assertTrue(history["ok"])
        self.assertEqual(history["workload_hash"], self.workload.workload_hash)
        self.assertEqual(len(history["summaries"]), 1)
        self.assertEqual(history["summaries"][0]["summary_id"], summary["summary_id"])

        # --- a second tune with a DIFFERENT provider (same knob grid,
        # different label) warm-starts from the first run's outcome without
        # crashing -- exact reordering is exercised in tests/test_tune.py.
        second_provider = DeclarativeProvider(
            "tune-cli-loop-provider-2",
            ({"mode": "eager", "tile": 16}, {"mode": "compiled", "tile": 24}),
        )
        second_provider_path = _write(self.base / "provider2.json", second_provider.to_dict())
        flags = self._context_flags()
        flags[flags.index(str(self.provider_path))] = str(second_provider_path)
        status, stdout, stderr = _run_cli("tune", *flags)
        self.assertEqual(status, 0, stderr)
        second_summary = json.loads(stdout)
        self.assertEqual(len(second_summary["entrants"]), 2)

        status, stdout, stderr = _run_cli(
            "history",
            "--workload", str(self.workload_path),
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
        )
        self.assertEqual(status, 0, stderr)
        self.assertEqual(len(json.loads(stdout)["summaries"]), 2)


# ---------------------------------------------------------------------------
# Fail-closed boundaries: run on every host, no MLX or real sandbox needed.
# ---------------------------------------------------------------------------


class TuneCLIFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.artifact_root = self.base / "artifacts"
        self.artifact_root.mkdir()
        self.store_root = self.base / "store"
        self.key_dir = self.base / "keys"
        self.workload = _toy_matmul_workload()
        self.provider = DeclarativeProvider("fail-closed-provider", ({"mode": "eager", "tile": 16},))
        self.workload_path = _write(self.base / "workload.json", self.workload.to_dict())
        self.provider_path = _write(self.base / "provider.json", self.provider.to_dict())

    def _context_flags(self) -> list[str]:
        return [
            "--workload", str(self.workload_path),
            "--provider", str(self.provider_path),
            "--artifact-root", str(self.artifact_root),
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
        ]

    def _empty_path(self) -> contextlib.AbstractContextManager:
        empty_path_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.rmdir(empty_path_dir))
        return patch.dict(os.environ, {"PATH": empty_path_dir})

    def test_tune_fails_closed_when_sandbox_primitives_are_unavailable(self) -> None:
        with self._empty_path():
            status, stdout, stderr = _run_cli("tune", *self._context_flags())
        self.assertEqual(status, EXIT_UNAVAILABLE)
        self.assertEqual(stdout, "")
        diagnostic = json.loads(stderr)
        self.assertEqual(diagnostic["error"]["code"], "unavailable")
        self.assertEqual(diagnostic["error"]["details"]["status"], "unavailable")
        self.assertEqual(diagnostic["error"]["details"]["stage"], "G1")
        self.assertEqual(diagnostic["error"]["details"]["surface"], "local_sandbox")

    def test_unknown_workload_name_is_a_typed_diagnostic(self) -> None:
        unknown_workload = FrozenWorkload(
            "not-a-known-workload",
            knobs=(Knob("mode", "enum", values=("eager", "compiled")),),
        )
        unknown_path = _write(self.base / "unknown_workload.json", unknown_workload.to_dict())
        unknown_provider = DeclarativeProvider("p", ({"mode": "eager"},))
        unknown_provider_path = _write(self.base / "unknown_provider.json", unknown_provider.to_dict())
        status, stdout, stderr = _run_cli(
            "tune",
            "--workload", str(unknown_path),
            "--provider", str(unknown_provider_path),
            "--artifact-root", str(self.artifact_root),
            "--store", str(self.store_root),
            "--key-dir", str(self.key_dir),
        )
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        diagnostic = json.loads(stderr)
        self.assertEqual(diagnostic["error"]["code"], "provider_error")
        self.assertIn("not-a-known-workload", diagnostic["error"]["message"])

    def test_history_never_needs_sandbox_primitives_and_is_empty_for_a_fresh_store(self) -> None:
        with self._empty_path():
            status, stdout, stderr = _run_cli(
                "history",
                "--workload", str(self.workload_path),
                "--store", str(self.store_root),
                "--key-dir", str(self.key_dir),
            )
        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        result = json.loads(stdout)
        self.assertTrue(result["ok"])
        self.assertEqual(result["summaries"], [])

    def test_tune_missing_provider_flag_is_a_usage_error(self) -> None:
        status, stdout, stderr = _run_cli(
            "tune",
            "--workload", str(self.workload_path),
            "--artifact-root", str(self.artifact_root),
        )
        self.assertNotEqual(status, 0)
        self.assertEqual(stdout, "")


if __name__ == "__main__":
    unittest.main()
