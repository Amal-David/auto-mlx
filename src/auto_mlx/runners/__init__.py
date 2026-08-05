"""Trusted runner scripts and their pure-stdlib wiring helpers.

Everything importable from this package (``auto_mlx.runners``, this module,
and ``auto_mlx.runners.reference``) is stdlib-only and safe to import
without MLX installed.  ``auto_mlx.runners.reference_matmul`` is the one
exception: it is a standalone script meant to be *launched as a subprocess*
(``[sys.executable, reference_matmul_script_path()]``), never imported --
importing it is still MLX-free (mlx is imported inside its functions, not
at module scope), but its whole purpose only makes sense when it runs
in its own process under the evaluator's trusted-runner contract.
"""

from __future__ import annotations

from typing import Final

from .reference import (
    BASELINE_RUNNER_ID,
    CANDIDATE_RUNNER_ID,
    reference_matmul_script_path,
    register_reference_matmul_runners,
)

__all__: Final = [
    "BASELINE_RUNNER_ID",
    "CANDIDATE_RUNNER_ID",
    "reference_matmul_script_path",
    "register_reference_matmul_runners",
]
