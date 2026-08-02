from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import subprocess
import stat
import tempfile
import tomllib
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import auto_mlx
import auto_mlx.cli as cli
from auto_mlx import CandidateProposal, EvaluationPolicy, FrozenWorkload, Knob, RuntimeIdentity, canonical_json
from auto_mlx.cli import EXIT_CONTRACT, EXIT_IO, EXIT_UNAVAILABLE, EXIT_USAGE, MAX_JSON_INPUT_BYTES, main
from auto_mlx.errors import Failure, FailureCode
from auto_mlx.receipts import RawSample, Receipt


class CLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = FrozenWorkload(
            "cli-toy",
            knobs=(
                Knob("mode", "enum", values=("eager", "compiled")),
                Knob("tile", "integer", minimum=16, maximum=32),
            ),
            parameters={"shape": [1, 128, 128], "dtype": "float16"},
        )

    def _run(self, *arguments: str, stdin: str | None = None) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if stdin is None:
                status = main(list(arguments))
            else:
                previous_stdin = sys.stdin
                sys.stdin = io.StringIO(stdin)
                try:
                    status = main(list(arguments))
                finally:
                    sys.stdin = previous_stdin
        return status, stdout.getvalue(), stderr.getvalue()

    def _write(self, directory: Path, name: str, value: object) -> Path:
        path = directory / name
        path.write_text(canonical_json(value), encoding="utf-8")
        return path

    def test_validate_workload_is_json_and_does_not_need_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = self._write(directory, "workload.json", self.workload.to_dict())
            status, stdout, stderr = self._run("validate", "workload", "--input", str(path))
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        result = json.loads(stdout)
        self.assertTrue(result["ok"])
        self.assertEqual(result["kind"], "workload")
        self.assertEqual(result["document"]["name"], "cli-toy")

    def test_inspect_candidate_reports_derived_identity(self) -> None:
        proposal = CandidateProposal("grid", self.workload, {"mode": "eager", "tile": 16})
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            workload_path = self._write(directory, "workload.json", self.workload.to_dict())
            candidate_path = self._write(directory, "candidate.json", proposal.to_dict())
            status, stdout, stderr = self._run(
                "inspect",
                "candidate",
                str(candidate_path),
                "--workload",
                str(workload_path),
            )
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["ids"]["candidate_id"], proposal.candidate_id)

    def test_inspect_receipt_validates_its_immutable_identity_without_mlx(self) -> None:
        proposal = CandidateProposal("grid", self.workload, {"mode": "eager", "tile": 16})
        receipt = Receipt(
            self.workload,
            proposal,
            EvaluationPolicy(measurement_runs=1),
            RuntimeIdentity("python", "3.11.0", "Darwin", "arm64"),
            [RawSample(0, 12, 14, {"token": 1}, {"token": 1}, 0)],
            created_at_ns=1,
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            path = self._write(Path(raw_directory), "receipt.json", receipt.to_dict())
            status, stdout, stderr = self._run("inspect", "receipt", str(path))
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["ids"]["receipt_id"], receipt.receipt_id)

    def test_bad_contract_is_stderr_only_with_stable_nonzero_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = self._write(Path(raw_directory), "bad.json", {"name": "missing-fields"})
            status, stdout, stderr = self._run("validate", "workload", str(path))
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        diagnostic = json.loads(stderr)
        self.assertFalse(diagnostic["ok"])
        self.assertEqual(diagnostic["error"]["code"], "invalid_value")

    def test_surrogate_object_key_has_stable_cli_diagnostic_and_exit_code(self) -> None:
        raw = '{"name":"cli-toy","artifacts":[],"knobs":[],"parameters":{"\\ud800":1}}'
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "surrogate-key.json"
            path.write_text(raw, encoding="ascii")
            status, stdout, stderr = self._run("validate", "workload", str(path))
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        diagnostic = json.loads(stderr)
        self.assertFalse(diagnostic["ok"])
        self.assertEqual(diagnostic["error"]["code"], FailureCode.INVALID_UNICODE.value)
        self.assertNotIn("\ud800", diagnostic["error"]["message"])

    def test_missing_input_is_usage_error(self) -> None:
        status, stdout, stderr = self._run("validate", "workload")
        self.assertEqual(status, EXIT_USAGE)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "usage_error")

    def test_oversized_file_is_rejected_before_json_parsing(self) -> None:
        payload = b'{"value":"' + (b"x" * MAX_JSON_INPUT_BYTES) + b'"}'
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "oversized.json"
            path.write_bytes(payload)
            status, stdout, stderr = self._run("validate", "document", str(path))
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        diagnostic = json.loads(stderr)
        self.assertEqual(diagnostic["error"]["code"], FailureCode.INPUT_TOO_LARGE.value)
        self.assertIn(str(MAX_JSON_INPUT_BYTES), diagnostic["error"]["message"])

    def test_oversized_stdin_is_rejected_before_json_parsing(self) -> None:
        oversized = '{"value":"' + ("x" * MAX_JSON_INPUT_BYTES) + '"}'
        status, stdout, stderr = self._run("validate", "document", "-", stdin=oversized)
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], FailureCode.INPUT_TOO_LARGE.value)

    def test_deep_nesting_is_a_contract_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "deep.json"
            path.write_text("[" * 5000 + "0" + "]" * 5000, encoding="utf-8")
            status, stdout, stderr = self._run("validate", "document", str(path))
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "invalid_json")

    def test_unpaired_surrogate_is_a_contract_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "surrogate.json"
            path.write_bytes(b'{"value":"\\ud800"}')
            status, stdout, stderr = self._run("validate", "document", str(path))
        self.assertEqual(status, EXIT_CONTRACT)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "invalid_json")

    def test_stdin_remains_a_supported_input_source(self) -> None:
        status, stdout, stderr = self._run("validate", "document", "-", stdin='{"value":1}')
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["document"], {"value": 1})

    def test_non_regular_input_is_rejected_before_any_read(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is not available")
        with tempfile.TemporaryDirectory() as raw_directory:
            fifo = Path(raw_directory) / "input.fifo"
            os.mkfifo(fifo)
            status, stdout, stderr = self._run("validate", "document", str(fifo))
        self.assertEqual(status, EXIT_IO)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "io_error")

    def test_symlink_input_is_rejected_by_open_without_following_it(self) -> None:
        if not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"):
            self.skipTest("symlink-safe input primitives are not available")
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            target = self._write(directory, "target.json", {"value": 1})
            link = directory / "input.json"
            link.symlink_to(target.name)
            status, stdout, stderr = self._run("validate", "document", str(link))
        self.assertEqual(status, EXIT_IO)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "io_error")

    def test_input_is_read_from_the_open_descriptor_after_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            path = self._write(directory, "input.json", {"value": 1})
            original_open = os.open
            opened = False

            def open_and_swap(value: object, flags: int, mode: int = 0o777, **kwargs: object) -> int:
                nonlocal opened
                descriptor = original_open(value, flags, mode, **kwargs)
                if not opened and Path(value) == path:
                    replacement = path.with_name("replacement.json")
                    path.rename(replacement)
                    path.write_text('{"value":2}', encoding="utf-8")
                    opened = True
                return descriptor

            with mock.patch("auto_mlx.cli.os.open", side_effect=open_and_swap):
                status, stdout, stderr = self._run("validate", "document", str(path))
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["document"], {"value": 1})

    def test_ambiguous_long_options_are_rejected(self) -> None:
        status, stdout, stderr = self._run("validate", "document", "--inp", "-")
        self.assertEqual(status, EXIT_USAGE)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "usage_error")

    def test_irrelevant_output_and_context_options_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            source = self._write(Path(raw_directory), "document.json", {"value": 1})
            cases = (
                ("inspect", "document", str(source), "--output", str(Path(raw_directory) / "out.json")),
                ("validate", "document", str(source), "--workload", str(source)),
                ("validate", "provider", str(source), "--artifact-root", raw_directory),
            )
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    status, stdout, stderr = self._run(*arguments)
                    self.assertEqual(status, EXIT_USAGE)
                    self.assertEqual(stdout, "")
                    self.assertEqual(json.loads(stderr)["error"]["code"], "usage_error")

    def test_failure_default_details_is_python311_compatible_and_immutable(self) -> None:
        first = Failure(FailureCode.INVALID_VALUE, "bad")
        second = Failure(FailureCode.INVALID_VALUE, "also bad")
        self.assertEqual(first.details, {})
        self.assertIsNot(first.details, second.details)
        with self.assertRaises(TypeError):
            first.details["extra"] = True  # type: ignore[index]

    def test_python311_subprocess_can_import_errors_and_cli_when_available(self) -> None:
        executable = shutil.which("python3.11")
        if executable is None:
            self.skipTest("python3.11 is not available")
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        completed = subprocess.run(
            [
                executable,
                "-c",
                "from auto_mlx.errors import Failure, FailureCode; from auto_mlx import cli; Failure(FailureCode.INVALID_VALUE, 'ok'); print(cli.MAX_JSON_INPUT_BYTES)",
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), str(MAX_JSON_INPUT_BYTES))

    def test_deferred_commands_fail_closed_and_do_not_claim_success(self) -> None:
        expected_stages = {"evaluate": "G1", "promote": "G2", "dispatch": "G2"}
        for command in ("evaluate", "promote", "dispatch"):
            with self.subTest(command=command):
                status, stdout, stderr = self._run(command)
                self.assertEqual(status, EXIT_UNAVAILABLE)
                self.assertEqual(stdout, "")
                diagnostic = json.loads(stderr)
                self.assertEqual(diagnostic["error"]["code"], "unavailable")
                self.assertEqual(diagnostic["error"]["details"]["status"], "deferred")
                self.assertEqual(diagnostic["error"]["details"]["stage"], expected_stages[command])
                self.assertEqual(diagnostic["error"]["details"]["surface"], "cli_orchestration")
                self.assertIn("library exists", diagnostic["error"]["message"])

    def test_help_describes_deferred_cli_wiring_without_stale_library_claims(self) -> None:
        status, stdout, stderr = self._run("--help")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("CLI orchestration is deferred", stdout)
        self.assertIn("evaluator", stdout)
        self.assertIn("library exists", stdout)
        self.assertNotIn("not implemented", stdout)

    def test_documented_example_commands_are_fresh_checkout_safe(self) -> None:
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        examples = (root / "examples" / "README.md").read_text(encoding="utf-8")
        self.assertIn("PYTHONPATH=src python3 -m auto_mlx validate provider", readme)
        self.assertIn("PYTHONPATH=src python3 -m auto_mlx validate workload", readme)
        self.assertIn("PYTHONPATH=src python3 -m auto_mlx validate workload", examples)
        self.assertIn("PYTHONPATH=src python3 -m auto_mlx inspect provider", examples)

        status, stdout, stderr = self._run(
            "validate", "workload", "--input", str(root / "examples" / "workload.json")
        )
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertTrue(json.loads(stdout)["ok"])

    def test_source_tree_version_does_not_prefer_stale_installed_metadata(self) -> None:
        with mock.patch("auto_mlx.cli.distribution_version", return_value="99.99.99"):
            status, stdout, stderr = self._run("--version")
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout.strip(), "auto-mlx 0.1.0")

    def test_ci_runs_from_standalone_repository_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertNotIn("working-directory: auto-mlx", workflow)
        self.assertIn("run: python -m py_compile src/auto_mlx/*.py tests/*.py", workflow)
        self.assertIn("run: python -m unittest discover -s tests -p 'test_*.py' -v", workflow)
        self.assertIn("os: [ubuntu-latest, macos-latest]", workflow)
        self.assertIn('python-version: ["3.11", "3.14"]', workflow)
        self.assertIn("--workdir", workflow)

    def test_stable_lane_apis_are_exported_from_package_root(self) -> None:
        from auto_mlx.dispatch import DispatchResult, dispatch
        from auto_mlx.evaluator import Evaluator
        from auto_mlx.promotion import PromotionDecision, make_promotion_decision
        from auto_mlx.receipts import Receipt, validate_receipt

        self.assertIs(auto_mlx.Evaluator, Evaluator)
        self.assertIs(auto_mlx.Receipt, Receipt)
        self.assertIs(auto_mlx.PromotionDecision, PromotionDecision)
        self.assertIs(auto_mlx.DispatchResult, DispatchResult)
        self.assertIs(auto_mlx.make_promotion_decision, make_promotion_decision)
        self.assertIs(auto_mlx.dispatch, dispatch)
        self.assertIs(auto_mlx.validate_receipt, validate_receipt)

    def test_schema_packaging_metadata_remains_declared(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        data_files = config["tool"]["setuptools"]["data-files"]
        self.assertEqual(data_files["schemas"], ["schemas/*.json"])
        self.assertEqual(config["project"]["license"], "MIT")

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write(directory, "workload.json", self.workload.to_dict())
            output = directory / "canonical.json"
            first = self._run("validate", "workload", str(source), "--output", str(output))
            second = self._run("validate", "workload", str(source), "--output", str(output))
        self.assertEqual(first[0], 0)
        self.assertEqual(second[0], EXIT_IO)
        self.assertEqual(second[1], "")
        self.assertEqual(json.loads(second[2])["error"]["code"], "io_error")
        self.assertIn("refusing to overwrite", second[2])

    def test_output_is_private_create_only_and_synced(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write(directory, "workload.json", self.workload.to_dict())
            output = directory / "canonical.json"
            status, stdout, stderr = self._run("validate", "workload", str(source), "--output", str(output))
            self.assertEqual(status, 0)
            self.assertEqual(stderr, "")
            self.assertEqual(json.loads(stdout)["output"], str(output))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(output.read_text(encoding="utf-8"), canonical_json(self.workload.to_dict()))

    def test_output_write_failure_reports_created_destination_without_temp_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write(directory, "workload.json", self.workload.to_dict())
            output = directory / "canonical.json"
            def fail_file_fsync(descriptor: int) -> None:
                raise OSError("simulated fsync failure")

            with mock.patch("auto_mlx.cli.os.fsync", side_effect=fail_file_fsync):
                status, stdout, stderr = self._run("validate", "workload", str(source), "--output", str(output))
            self.assertEqual(status, EXIT_IO)
            self.assertEqual(stdout, "")
            diagnostic = json.loads(stderr)
            self.assertEqual(diagnostic["error"]["code"], "io_error")
            self.assertIn("contents were not durably synced", diagnostic["error"]["message"])
            self.assertTrue(output.exists())
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_directory_fsync_failure_is_distinct_and_leaves_output_present(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write(directory, "workload.json", self.workload.to_dict())
            output = directory / "canonical.json"
            original_fsync = os.fsync
            calls = 0

            def fail_directory_fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated directory fsync failure")
                original_fsync(descriptor)

            with mock.patch("auto_mlx.cli.os.fsync", side_effect=fail_directory_fsync):
                status, stdout, stderr = self._run("validate", "workload", str(source), "--output", str(output))
            self.assertEqual(status, EXIT_IO)
            self.assertEqual(stdout, "")
            message = json.loads(stderr)["error"]["message"]
            self.assertIn("directory durability is unconfirmed", message)
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_text(encoding="utf-8"), canonical_json(self.workload.to_dict()))
            self.assertEqual(list(directory.glob(".*")), [])

    def test_fchmod_failure_closes_output_descriptor_and_reports_created_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write(directory, "workload.json", self.workload.to_dict())
            output = directory / "canonical.json"
            fchmod_descriptor: int | None = None

            def fail_fchmod(descriptor: int, mode: int) -> None:
                nonlocal fchmod_descriptor
                fchmod_descriptor = descriptor
                raise OSError("simulated fchmod failure")

            with mock.patch("auto_mlx.cli.os.fchmod", side_effect=fail_fchmod), mock.patch(
                "auto_mlx.cli.os.close", wraps=os.close
            ) as close:
                status, stdout, stderr = self._run("validate", "workload", str(source), "--output", str(output))
            self.assertEqual(status, EXIT_IO)
            self.assertEqual(stdout, "")
            self.assertIn("private permissions could not be confirmed", json.loads(stderr)["error"]["message"])
            self.assertIsNotNone(fchmod_descriptor)
            close.assert_any_call(fchmod_descriptor)
            self.assertTrue(output.exists())

    def test_output_destination_symlink_is_not_followed_or_replaced(self) -> None:
        if not hasattr(os, "symlink") or not hasattr(os, "O_NOFOLLOW"):
            self.skipTest("symlink-safe output primitives are not available")
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write(directory, "workload.json", self.workload.to_dict())
            target = directory / "target.json"
            target.write_text("keep", encoding="utf-8")
            output = directory / "canonical.json"
            output.symlink_to(target.name)
            status, stdout, stderr = self._run("validate", "workload", str(source), "--output", str(output))
            self.assertEqual(status, EXIT_IO)
            self.assertEqual(stdout, "")
            self.assertIn("refusing to overwrite", json.loads(stderr)["error"]["message"])
            self.assertTrue(output.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_output_uses_opened_parent_descriptor_if_parent_path_is_swapped(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            parent = root / "parent"
            moved = root / "moved"
            parent.mkdir()
            output = parent / "canonical.json"
            original_open = os.open
            swapped = False

            def open_and_swap(path: object, flags: int, mode: int = 0o777, **kwargs: object) -> int:
                nonlocal swapped
                descriptor = original_open(path, flags, mode, **kwargs)
                if not swapped and kwargs.get("dir_fd") is None and Path(path) == parent:
                    parent.rename(moved)
                    parent.mkdir()
                    swapped = True
                return descriptor

            with mock.patch("auto_mlx.cli.os.open", side_effect=open_and_swap):
                cli._create_only(str(output), b'{"value":1}')
            self.assertTrue((moved / "canonical.json").is_file())
            self.assertFalse((parent / "canonical.json").exists())

    def test_missing_output_parent_is_an_io_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            source = self._write(directory, "document.json", {"value": 1})
            output = directory / "missing" / "canonical.json"
            status, stdout, stderr = self._run("validate", "document", str(source), "--output", str(output))
        self.assertEqual(status, EXIT_IO)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(stderr)["error"]["code"], "io_error")

    def test_broken_stdout_is_a_handled_io_failure(self) -> None:
        class BrokenPipeStream:
            def write(self, value: str) -> int:
                raise BrokenPipeError("closed pipe")

            def flush(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as raw_directory:
            source = self._write(Path(raw_directory), "document.json", {"value": 1})
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(BrokenPipeStream()):
                status = main(["validate", "document", str(source)])
        self.assertEqual(status, EXIT_IO)
        self.assertEqual(json.loads(stderr.getvalue())["error"]["code"], "io_error")


if __name__ == "__main__":
    unittest.main()
