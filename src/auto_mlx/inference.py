"""Resident, network-denied MLX-LM worker with immutable model staging.

The parent remains MLX-free. Cancellation or timeout kills the worker and frees
its process resources; the next request can cold-reload the same verified model.
This is a single-user local runtime, not a multi-tenant sandbox.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time

from .canonical import canonical_bytes, canonical_json, strict_json_loads
from .errors import ContractError, FailureCode
from .model_bundle import ModelBundle, integer, require
from .paths import _open_verified_file
from .runners.mlx_lm import runner_path
from .runners.mlx_lm_runner import validate_request, config_value
from .sandbox import LocalSandboxProvider, LocalSandboxAuthority

MAX_EVENT_BYTES = 262144
MAX_RESPONSE_BYTES = 4 * 1024**2


class InferenceSession:
    def __init__(self, bundle: ModelBundle, model_root: Path, *, timeout_seconds: int = 120):
        integer(timeout_seconds, 1, 600, "timeout_seconds")
        self.bundle = bundle
        self.model_root = Path(model_root)
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._temporary = None
        self._process = None
        self._selector = None
        self._stderr = None
        self._buffer = b""
        self.load_elapsed_ns = None
        self._closed = False

    def _snapshot(self, target: Path) -> None:
        self.bundle.verify(self.model_root, check_runtime=True)
        target.mkdir(mode=0o700)
        for artifact in self.bundle.artifacts:
            # The private snapshot, not the checked source path, is what the
            # native loader sees. Every copied byte is checked on its open fd.
            descriptor = _open_verified_file(str(self.model_root), artifact.path)
            digest = hashlib.sha256()
            count = 0
            try:
                with (target / artifact.path).open("xb") as destination:
                    while count < artifact.size_bytes:
                        chunk = os.read(descriptor, min(1024**2, artifact.size_bytes - count))
                        require(bool(chunk), "model artifact shortened during staging")
                        destination.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
                    require(not os.read(descriptor, 1), "model artifact grew during staging")
            finally:
                os.close(descriptor)
            require(digest.hexdigest() == artifact.sha256, "model artifact changed during staging")
            os.chmod(target / artifact.path, 0o400)

    def _read_event(self, deadline: float) -> dict:
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ContractError("native worker exceeded its deadline", code=FailureCode.TIMEOUT)
            if not self._selector.select(remaining):
                raise ContractError("native worker exceeded its deadline", code=FailureCode.TIMEOUT)
            chunk = os.read(self._process.stdout.fileno(), 65536)
            require(bool(chunk), "native worker exited before completing its response")
            self._buffer += chunk
            require(len(self._buffer.split(b"\n", 1)[0]) <= MAX_EVENT_BYTES, "native worker exceeded event byte limit")
        raw, self._buffer = self._buffer.split(b"\n", 1)
        value = strict_json_loads(raw)
        require(type(value) is dict and value.get("event") in {"ready", "delta", "done", "error"}, "invalid native worker event")
        return value

    def _start(self) -> None:
        require(not self._closed, "inference session is closed")
        if self._process is not None and self._process.poll() is None:
            return
        self._stop_worker()
        started = time.perf_counter_ns()
        self._temporary = tempfile.TemporaryDirectory(prefix="auto-mlx-native-")
        root = Path(self._temporary.name).resolve()
        try:
            model = root / "model"
            self._snapshot(model)
            self._stderr = (root / "worker-stderr.log").open("wb")
            argv = (sys.executable, "-I", str(runner_path()), "--worker", "--spec-json=" + canonical_json(self.bundle.to_dict()), "--python-site=" + sysconfig.get_paths()["purelib"])
            env = {"PATH": "/usr/bin:/bin", "HOME": str(root), "TMPDIR": str(root), "TEMP": str(root), "TMP": str(root), "PYTHONDONTWRITEBYTECODE": "1", "AUTO_MLX_ARTIFACT_ROOT": str(model)}
            provider = LocalSandboxProvider(max_open_files=4096, cpu_seconds=600)
            isolated = provider.enforce(argv, cwd=str(root), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr)
            self._process = isolated.process
            LocalSandboxAuthority().verify(provider, self._process, isolated.claim)
            self._selector = selectors.DefaultSelector()
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            event = self._read_event(time.monotonic() + self.timeout_seconds)
            require(event.get("event") == "ready", event.get("message", "native worker did not become ready"))
            self.load_elapsed_ns = time.perf_counter_ns() - started
        except BaseException:
            self._stop_worker()
            raise

    def warmup(self) -> None:
        with self._lock:
            self._start()

    def stream(self, request: dict, *, prefill_step_size: int = 2048):
        try:
            validate_request(request)
            config_value({"prefill_step_size": prefill_step_size})
        except ValueError as exc:
            raise ContractError(str(exc), code=FailureCode.INVALID_VALUE) from exc
        require(self._lock.acquire(blocking=False), "native worker is busy")
        complete = False
        try:
            self._start()
            wire = canonical_bytes({**request, "prefill_step_size": prefill_step_size}) + b"\n"
            self._process.stdin.write(wire)
            self._process.stdin.flush()
            deadline = time.monotonic() + self.timeout_seconds
            total = 0
            deltas = 0
            while True:
                event = self._read_event(deadline)
                kind = event["event"]
                if kind == "error":
                    raise ContractError(event.get("message", "native worker failed"), code=FailureCode.RUNTIME_FAILURE)
                require(kind in {"delta", "done"}, "unexpected native worker event")
                total += len(canonical_bytes(event))
                require(total <= MAX_RESPONSE_BYTES, "native worker exceeded response byte limit")
                if kind == "delta":
                    deltas += 1
                    require(deltas <= request["max_tokens"], "native worker exceeded token limit")
                else:
                    complete = True
                    event = {**event, "model_load_ns": self.load_elapsed_ns, "bundle_id": self.bundle.bundle_id}
                yield event
                if complete:
                    break
        finally:
            if not complete:
                # Closing an abandoned iterator, client disconnect, and deadline
                # expiry all terminate the generation, rather than letting it run.
                self._stop_worker()
            self._lock.release()

    def generate(self, request: dict, *, prefill_step_size: int = 2048) -> dict:
        result = None
        for event in self.stream(request, prefill_step_size=prefill_step_size):
            if event["event"] == "done":
                result = event
        require(result is not None, "native worker produced no result")
        return result

    def _stop_worker(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    process.kill()
            try:
                process.wait(timeout=5)
            finally:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._buffer = b""

    def close(self) -> None:
        # Call only after the request iterator is exhausted or explicitly closed.
        with self._lock:
            self._stop_worker()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
