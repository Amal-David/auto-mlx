"""Opt-in real-checkpoint tests. Never downloads weights or enables remote code."""
from __future__ import annotations
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from auto_mlx.model_bundle import prepare_bundle, inspect_model_files
from auto_mlx.canonical import canonical_bytes
from auto_mlx.inference import InferenceSession
from auto_mlx.inference_server import LocalInferenceServer
from auto_mlx.errors import ContractError


@unittest.skipUnless(os.environ.get("AUTO_MLX_NATIVE_MODEL"), "set AUTO_MLX_NATIVE_MODEL to an ordinary local native checkpoint directory")
class NativeModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["AUTO_MLX_NATIVE_MODEL"]).resolve(strict=True)
        _, artifacts = inspect_model_files(cls.root)
        revision = hashlib.sha256(canonical_bytes([a.to_dict() for a in artifacts])).hexdigest()
        cls.request = {"prompt": "The capital of France is", "max_tokens": 8}
        cls.bundle = prepare_bundle(cls.root, source={"id": "local-test-input", "revision": revision, "license": "operator-supplied-test-input"}, fixtures=[cls.request], context_limit=64, device=os.environ.get("AUTO_MLX_NATIVE_DEVICE", "cpu"))

    def test_repeatability_context_limit_cancel_and_reload(self):
        with InferenceSession(self.bundle, self.root, timeout_seconds=90) as session:
            first = session.generate(self.request)
            second = session.generate(self.request)
            for key in ("text", "token_ids", "finish_reason"):
                self.assertEqual(first[key], second[key])
            self.assertGreater(first["time_to_first_token_ns"], 0)
            self.assertLessEqual(first["completion_tokens"], 8)
            with self.assertRaisesRegex(ContractError, "context limit"):
                session.generate({"prompt": "hello " * 200, "max_tokens": 8})
            self.assertIsNone(session._process)
            stream = session.stream(self.request)
            self.assertEqual(next(stream)["event"], "delta")
            stream.close()
            self.assertIsNone(session._process)
            again = session.generate(self.request)
            self.assertEqual(first["token_ids"], again["token_ids"])

    def test_real_completion_and_sse_api(self):
        with InferenceSession(self.bundle, self.root, timeout_seconds=90) as session:
            session.warmup()
            server = LocalInferenceServer(0, session, lambda request: {"mode": "native_fallback", "prefill_step_size": 2048})
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for streaming in (False, True):
                    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=90)
                    try:
                        body = {**self.request, "model": self.bundle.bundle_id, "stream": streaming}
                        connection.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
                        response = connection.getresponse()
                        data = response.read()
                        self.assertEqual(response.status, 200, data)
                        if streaming:
                            self.assertTrue(data.endswith(b"data: [DONE]\n\n"), data)
                        else:
                            self.assertTrue(json.loads(data)["choices"][0]["text"])
                    finally:
                        connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
