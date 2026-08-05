"""Pure-stdlib wiring for the reference MLX runner (``reference_matmul.py``).

This module never imports ``mlx`` and never imports
``auto_mlx.runners.reference_matmul`` as Python -- it only ever refers to
that script's *path*, which becomes an ``argv``/artifact entry that a
``TrustedRunner`` binds and verifies by content hash.  MLX only ever runs
inside the subprocess that ``execute_plan`` launches, never in this
process.

This is the library-level helper a future CLI wave wires up; it is
otherwise unused today except by ``tests/test_runner_reference.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from ..executor import TrustedRunner, TrustedRunnerRegistry


BASELINE_RUNNER_ID: Final = "reference-matmul-baseline"
CANDIDATE_RUNNER_ID: Final = "reference-matmul-candidate"

# The runner only ever forces the baseline arm to "eager"; every other mode
# value is left to the candidate's own (shared) config.  See
# reference_matmul.main's --force-mode handling.
_BASELINE_FORCED_MODE: Final = "eager"


def reference_matmul_script_path() -> Path:
    """Absolute, resolved path to the standalone ``reference_matmul.py`` script."""

    return (Path(__file__).resolve().parent / "reference_matmul.py").resolve(strict=True)


def register_reference_matmul_runners(
    registry: TrustedRunnerRegistry,
    *,
    python_executable: str | None = None,
    baseline_runner_id: str = BASELINE_RUNNER_ID,
    candidate_runner_id: str = CANDIDATE_RUNNER_ID,
) -> tuple[TrustedRunner, TrustedRunner]:
    """Register the baseline and candidate reference-matmul runners.

    Both runner ids point at the exact same interpreter and script bytes
    (``reference_matmul.py`` is entirely config-driven via
    ``AUTO_MLX_CONFIG_PATH``); an ``Evaluator`` binds the same
    ``CandidateProposal`` config to both its baseline and candidate
    execution plans, so the two arms are differentiated purely by their
    ``argv``, which is itself part of each ``TrustedRunner``'s verified,
    immutable digest:

    - ``baseline_runner_id``: argv carries ``--force-mode=eager``, so the
      baseline arm always runs eager regardless of what the shared config's
      ``mode`` says.
    - ``candidate_runner_id``: no override; the candidate arm runs whatever
      ``mode`` the proposal's config declares (e.g. ``"compiled"``).

    Returns ``(baseline_runner, candidate_runner)``.
    """

    if not isinstance(registry, TrustedRunnerRegistry):
        raise TypeError("registry must be a TrustedRunnerRegistry")
    executable = python_executable if python_executable is not None else sys.executable
    interpreter = str(Path(executable).resolve(strict=True))
    script = str(reference_matmul_script_path())
    artifact_paths = (interpreter, script)
    baseline = registry.register_command(
        baseline_runner_id,
        (interpreter, script, f"--force-mode={_BASELINE_FORCED_MODE}"),
        artifact_paths=artifact_paths,
    )
    candidate = registry.register_command(
        candidate_runner_id,
        (interpreter, script),
        artifact_paths=artifact_paths,
    )
    return baseline, candidate


__all__: Final = [
    "BASELINE_RUNNER_ID",
    "CANDIDATE_RUNNER_ID",
    "reference_matmul_script_path",
    "register_reference_matmul_runners",
]
