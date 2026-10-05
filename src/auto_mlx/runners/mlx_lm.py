"""Stdlib-only registration of the fixed, artifact-bound MLX-LM runner."""
from __future__ import annotations
import sys
import sysconfig
from pathlib import Path

from ..canonical import canonical_json
from ..contracts import FrozenWorkload
from ..executor import TrustedRunnerRegistry
from ..model_bundle import bundle_from_workload

BASELINE_ID = "mlx-lm-text-baseline-v1"
CANDIDATE_ID = "mlx-lm-text-candidate-v1"


def runner_path() -> Path:
    return Path(__file__).with_name("mlx_lm_runner.py").resolve(strict=True)


def register_mlx_lm_runners(workload: FrozenWorkload, registry: TrustedRunnerRegistry) -> tuple[str, str]:
    bundle = bundle_from_workload(workload)
    interpreter = str(Path(sys.executable).resolve(strict=True))
    script = str(runner_path())
    # TrustedRunner resolves interpreter symlinks. Pin the caller's venv site
    # directory in evaluator-owned argv so that resolution cannot lose the venv.
    argv = (interpreter, "-I", script, "--spec-json=" + canonical_json(bundle.to_dict()), "--python-site=" + sysconfig.get_paths()["purelib"])
    registry.register_command(BASELINE_ID, (*argv, "--force-baseline"), artifact_paths=(interpreter, script))
    registry.register_command(CANDIDATE_ID, argv, artifact_paths=(interpreter, script))
    return BASELINE_ID, CANDIDATE_ID
