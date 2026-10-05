from __future__ import annotations

import contextlib
import copy
import http.client
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_mlx import cli
from auto_mlx.canonical import canonical_bytes
from auto_mlx.contracts import FrozenWorkload
from auto_mlx.errors import AutoMLXError, ContractError
from auto_mlx.executor import TrustedRunnerRegistry
from auto_mlx.model_bundle import ModelBundle, prepare_bundle, bundle_from_workload
from auto_mlx.inference import InferenceSession
from auto_mlx.inference_cli import profile_resolver
from auto_mlx.inference_server import LocalInferenceServer
from auto_mlx.runners.mlx_lm import register_mlx_lm_runners
from auto_mlx.runners.mlx_lm_runner import loads, config_value, validate_request


class NativeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        self.write("config.json", {"model_type": "qwen2", "max_position_embeddings": 256})
        self.write("tokenizer_config.json", {"tokenizer_class": "Qwen2TokenizerFast"})
        self.write("tokenizer.json", {})
        (self.model / "model.safetensors").write_bytes(b"test-only-weight-bytes")
        self.source = {"id": "local/test", "revision": "a" * 40, "license": "apache-2.0"}
        self.fixtures = [{"prompt": "hello", "max_tokens": 4}]
        self.bundle = prepare_bundle(self.model, source=self.source, fixtures=self.fixtures)

    def write(self, name, value):
        (self.model / name).write_text(json.dumps(value))

    def test_bundle_is_deterministic_and_immutable(self):
        other = prepare_bundle(self.model, source=self.source, fixtures=self.fixtures)
        self.assertEqual(self.bundle.bundle_id, other.bundle_id)
        returned = self.bundle.to_dict()
        returned["source"]["id"] = "mutated"
        self.assertEqual(self.bundle.to_dict()["source"]["id"], "local/test")
        self.bundle.verify(self.model, check_runtime=True)

    def test_artifact_mutation_fails_verification(self):
        self.write("tokenizer.json", {"changed": True})
        with self.assertRaises(ContractError):
            self.bundle.verify(self.model)

    def test_unlisted_model_file_is_not_loaded(self):
        self.write("added_tokens.json", {})
        with self.assertRaises(ContractError):
            self.bundle.verify(self.model)

    def test_custom_model_code_is_rejected(self):
        self.write("config.json", {"model_type": "qwen2", "max_position_embeddings": 256, "auto_map": {"AutoModel": "custom.Model"}})
        with self.assertRaisesRegex(ContractError, "remote/custom"):
            prepare_bundle(self.model, source=self.source, fixtures=self.fixtures)

    def test_code_file_and_directory_are_rejected(self):
        for name, directory in (("custom.py", False), ("nested", True)):
            path = self.model / name
            path.mkdir() if directory else path.write_text("raise AssertionError('must never execute')")
            with self.assertRaises(ContractError):
                prepare_bundle(self.model, source=self.source, fixtures=self.fixtures)
            path.rmdir() if directory else path.unlink()

    def test_symlinked_weights_are_rejected(self):
        weight = self.model / "model.safetensors"
        weight.unlink()
        target = self.root / "outside"
        target.write_bytes(b"outside")
        weight.symlink_to(target)
        with self.assertRaises(ContractError):
            self.bundle.verify(self.model)

    def test_duplicate_nonfinite_and_external_metadata_fail(self):
        config = self.model / "config.json"
        for raw in ('{"model_type":"qwen2","model_type":"llama"}', '{"model_type":"qwen2","max_position_embeddings":NaN}', '{"model_type":"qwen2","max_position_embeddings":256,"weights_path":"../../outside"}'):
            config.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                prepare_bundle(self.model, source=self.source, fixtures=self.fixtures)

    def test_unknown_bundle_fields_and_bad_types_fail(self):
        for field, value in (("schema_version", True), ("context_limit", False), ("backend", "custom"), ("verified", True)):
            data = self.bundle.to_dict()
            data[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                ModelBundle.from_dict(data)

    def test_source_requires_immutable_revision(self):
        for rev in ("main", "latest", "a" * 39, ""):
            with self.subTest(revision=rev), self.assertRaises(ContractError):
                prepare_bundle(self.model, source={**self.source, "revision": rev}, fixtures=self.fixtures)

    def test_context_limit_cannot_be_inflated(self):
        data = self.bundle.to_dict()
        data["context_limit"] = 257
        with self.assertRaises(ContractError):
            ModelBundle.from_dict(data).verify(self.model)

    def test_weight_index_must_match_shards(self):
        self.write("model.safetensors.index.json", {"weight_map": {"x": "outside.safetensors"}})
        with self.assertRaisesRegex(ContractError, "shards"):
            prepare_bundle(self.model, source=self.source, fixtures=self.fixtures)

    def test_runtime_change_blocks_execution_and_workload(self):
        data = self.bundle.to_dict()
        data["runtime"]["mlx"] = "different"
        bundle = ModelBundle.from_dict(data)
        bundle.verify(self.model)
        with self.assertRaises(ContractError):
            bundle.verify(self.model, check_runtime=True)
        with self.assertRaises(ContractError):
            bundle_from_workload(bundle.workload())

    def test_workload_and_fixed_runner_binding(self):
        workload = self.bundle.workload()
        self.assertEqual(bundle_from_workload(workload), self.bundle)
        registry = TrustedRunnerRegistry()
        baseline, candidate = register_mlx_lm_runners(workload, registry)
        self.assertIn("--force-baseline", registry.resolve(baseline).argv)
        self.assertNotIn("--force-baseline", registry.resolve(candidate).argv)
        self.assertIn("-I", registry.resolve(candidate).argv)
        self.assertNotEqual(registry.resolve(baseline).digest, registry.resolve(candidate).digest)

    def test_workload_cannot_redefine_baseline_or_artifacts(self):
        for mutation in ("baseline", "artifacts", "knobs"):
            data = self.bundle.workload().to_dict()
            if mutation == "baseline":
                data["parameters"]["baseline_prefill_step_size"] = 128
            elif mutation == "artifacts":
                data["artifacts"] = []
            else:
                data["knobs"][0]["maximum"] = 100000
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                bundle_from_workload(FrozenWorkload.from_dict(data))

    def test_snapshot_is_independent_and_source_remains_unchanged(self):
        with InferenceSession(self.bundle, self.model) as session:
            target = self.root / "snapshot"
            session._snapshot(target)
            self.assertEqual((target / "model.safetensors").read_bytes(), b"test-only-weight-bytes")
            self.assertEqual((self.model / "model.safetensors").read_bytes(), b"test-only-weight-bytes")

    def test_cli_prepare_and_dry_run_without_model_imports(self):
        fixtures = self.root / "fixtures.json"
        fixtures.write_text(json.dumps(self.fixtures))
        output = self.root / "bundle.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["bundle", "prepare", "--model", str(self.model), "--source-id", "local/test", "--revision", "a"*40, "--license", "apache-2.0", "--fixtures", str(fixtures), "--output", str(output)]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(cli.main(argv), 0, stderr.getvalue())
            self.assertEqual(cli.main(["generate", "--bundle", str(output), "--model-root", str(self.model), "--prompt", "hello"]), 0, stderr.getvalue())
            self.assertEqual(cli.main(argv), cli.EXIT_IO)
        lines = stdout.getvalue().splitlines()
        self.assertTrue(json.loads(lines[1])["dry_run"])

    def test_missing_sandbox_is_unavailable_without_execution(self):
        output = self.root / "bundle.json"
        output.write_bytes(self.bundle.payload)
        with mock.patch.object(cli, "local_sandbox_primitives_available", return_value=False), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["generate", "--bundle", str(output), "--model-root", str(self.model), "--prompt", "hello", "--execute"]), cli.EXIT_UNAVAILABLE)

    def test_no_candidate_means_native_without_store(self):
        args = SimpleNamespace(candidate=None, policy=None, store=None, key_dir=None)
        resolve = profile_resolver(self.bundle, self.model, args)
        self.assertEqual(resolve(self.fixtures[0])["prefill_step_size"], 2048)


class NativeProtocolTests(unittest.TestCase):
    def test_protocol_rejects_duplicates_floats_and_code_fields(self):
        for raw in ('{"x":1,"x":2}', '{"x":0.1}', '{"x":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                loads(raw)
        for config in ({"prefill_step_size": True}, {"prefill_step_size": 127}, {"prefill_step_size": 4097}, {"prefill_step_size": 512, "command": "anything"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config_value(config)
        self.assertEqual(config_value({"prefill_step_size": 512}), 512)

    def test_request_bounds_are_strict(self):
        for request in ({"prompt": "", "max_tokens": 1}, {"prompt": "x", "max_tokens": True}, {"prompt": "x", "max_tokens": 0}, {"prompt": "x"*65537, "max_tokens": 1}, {"prompt": "x", "max_tokens": 1, "path": "outside"}):
            with self.subTest(request=str(request)[:100]), self.assertRaises(ValueError):
                validate_request(request)

    def test_abandoned_stream_terminates_worker(self):
        session = InferenceSession(mock.Mock(), Path("."))
        session._process = SimpleNamespace(stdin=io.BytesIO())
        with mock.patch.object(session, "_start"), mock.patch.object(session, "_stop_worker") as stop, mock.patch.object(session, "_read_event", return_value={"event": "delta", "text": "x", "token": 1}):
            stream = session.stream({"prompt": "hello", "max_tokens": 4})
            next(stream)
            stream.close()
            stop.assert_called_once()
            self.assertFalse(session._lock.locked())

    def test_worker_deadline_is_bounded(self):
        session = InferenceSession(mock.Mock(), Path("."))
        session._selector = mock.Mock()
        session._selector.select.return_value = []
        with self.assertRaisesRegex(ContractError, "deadline"):
            session._read_event(time.monotonic() + 0.01)

    def test_public_imports_do_not_import_mlx(self):
        source = Path(__file__).resolve().parents[1] / "src"
        code = f"import sys;sys.path.insert(0,{str(source)!r});import auto_mlx.cli,auto_mlx.inference,auto_mlx.inference_server,auto_mlx.runners.mlx_lm_runner;assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)"
        completed = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class LocalAPITests(unittest.TestCase):
    def setUp(self):
        class FakeSession:
            bundle = SimpleNamespace(bundle_id="a"*64)
            def stream(self, request, **kwargs):
                yield {"event": "delta", "text": "hello", "token": 1}
                yield {"event": "done", "text": "hello", "finish_reason": "length", "prompt_tokens": 2, "completion_tokens": 1, "time_to_first_token_ns": 1, "generation_elapsed_ns": 2, "peak_memory_bytes": 3}
        self.server = LocalInferenceServer(0, FakeSession(), lambda request: {"mode": "native_fallback", "prefill_step_size": 2048})
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def call(self, body, headers=None, path="/v1/completions"):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request("POST", path, body if isinstance(body, str) else json.dumps(body), {"Content-Type": "application/json", **(headers or {})})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def request(self, **extra):
        return {"model": "a"*64, "prompt": "test", "max_tokens": 1, **extra}

    def test_complete_and_streaming_responses(self):
        status, body = self.call(self.request())
        self.assertEqual(status, 200, body)
        self.assertEqual(json.loads(body)["choices"][0]["text"], "hello")
        status, body = self.call(self.request(stream=True))
        self.assertEqual(status, 200, body)
        self.assertTrue(body.endswith(b"data: [DONE]\n\n"))
        self.assertEqual(body.count(b'"text": "hello"'), 1)

    def test_browser_and_dns_rebinding_requests_are_rejected(self):
        for headers in ({"Origin": "https://evil.example"}, {"Host": "evil.example"}):
            with self.subTest(headers=headers):
                status, _ = self.call(self.request(), headers)
                self.assertEqual(status, 400)

    def test_malformed_and_unsupported_requests_are_rejected(self):
        for body in (self.request(model="other"), self.request(stream=1), self.request(temperature=1), self.request(command="anything"), '{"model":"a","model":"b"}'):
            with self.subTest(body=body):
                status, _ = self.call(body)
                self.assertEqual(status, 400)
        self.assertEqual(self.call(self.request(), path="/v1/chat/completions")[0], 400)

    def test_server_cannot_bind_a_remote_address(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
