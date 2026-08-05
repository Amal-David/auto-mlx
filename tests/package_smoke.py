from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAMES = {
    "artifact.json",
    "candidate_proposal.json",
    "declarative_provider.json",
    "evaluation_policy.json",
    "frozen_workload.json",
    "knob.json",
    "runtime_identity.json",
}


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {command}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    if completed.stderr:
        raise AssertionError(f"command wrote diagnostics on success: {command}\nstderr={completed.stderr}")
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--console-script", required=True, type=Path)
    args = parser.parse_args()

    # This process runs with the wheel-installed venv's interpreter, so the
    # package -- and its schemas, which ship as auto_mlx.schemas resources
    # rather than loose files at the venv's data root -- is importable here.
    from auto_mlx.schemas import schema_names, schema_text

    installed_schema_names = set(schema_names())
    if installed_schema_names != SCHEMA_NAMES:
        raise AssertionError(f"installed schemas differ: {installed_schema_names}")
    for schema_name in sorted(SCHEMA_NAMES):
        payload = json.loads(schema_text(schema_name))
        if not isinstance(payload, dict) or payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise AssertionError(f"invalid installed schema payload: {schema_name}")

    # examples/ and docs/ are repo-only material and must not be installed
    # into the wheel's data root at all.
    data_root = Path(sysconfig.get_path("data"))
    for leaked in (data_root / "examples", data_root / "docs", data_root / "schemas"):
        if leaked.exists():
            raise AssertionError(f"repo-only or relocated material leaked into the installed wheel: {leaked}")

    distribution = metadata.distribution("auto-mlx")
    if distribution.requires not in (None, []):
        raise AssertionError(f"runtime dependencies were installed: {distribution.requires}")

    for command in ([sys.executable, "-m", "auto_mlx", "--version"], [str(args.console_script), "--version"]):
        version_result = _run(command, cwd=args.workdir)
        if version_result.stdout.strip() != "auto-mlx 0.1.0":
            raise AssertionError(f"unexpected version output: {version_result.stdout!r}")

    # examples/ ships in the sdist and the checkout, but never in the wheel;
    # read the fixture straight from the checkout this script lives in.
    example = PROJECT_ROOT / "examples" / "workload.json"
    if not example.is_file():
        raise AssertionError(f"example fixture is missing from the checkout: {example}")

    for command in (
        [sys.executable, "-m", "auto_mlx", "validate", "workload", "--input", str(example)],
        [str(args.console_script), "validate", "workload", "--input", str(example)],
    ):
        result = _run(command, cwd=args.workdir)
        payload = json.loads(result.stdout)
        if payload.get("ok") is not True or payload.get("kind") != "workload":
            raise AssertionError(f"unexpected validation payload: {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
