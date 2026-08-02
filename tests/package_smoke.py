from __future__ import annotations

import argparse
import json
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys
import sysconfig


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
    parser.add_argument("--example", required=True, type=Path)
    parser.add_argument("--console-script", required=True, type=Path)
    args = parser.parse_args()
    project_root = args.example.resolve().parents[1]

    distribution = metadata.distribution("auto-mlx")
    if distribution.requires not in (None, []):
        raise AssertionError(f"runtime dependencies were installed: {distribution.requires}")

    for command in ([sys.executable, "-m", "auto_mlx", "--version"], [str(args.console_script), "--version"]):
        version_result = _run(command, cwd=project_root)
        if version_result.stdout.strip() != "auto-mlx 0.1.0":
            raise AssertionError(f"unexpected version output: {version_result.stdout!r}")

    for command in (
        [sys.executable, "-m", "auto_mlx", "validate", "workload", "--input", str(args.example)],
        [str(args.console_script), "validate", "workload", "--input", str(args.example)],
    ):
        result = _run(command, cwd=project_root)
        payload = json.loads(result.stdout)
        if payload.get("ok") is not True or payload.get("kind") != "workload":
            raise AssertionError(f"unexpected validation payload: {payload}")

    schema_root = Path(sysconfig.get_path("data")) / "schemas"
    installed_schema_names = {path.name for path in schema_root.glob("*.json")}
    if installed_schema_names != SCHEMA_NAMES:
        raise AssertionError(f"installed schemas differ: {installed_schema_names}")
    for schema_name in sorted(SCHEMA_NAMES):
        with (schema_root / schema_name).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise AssertionError(f"invalid installed schema payload: {schema_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
