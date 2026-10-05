"""Fixed native MLX-LM subprocess. Never imports MLX or auto_mlx at module scope.

Only evaluator-owned argv selects a bundle. Candidate stdin/config contains
bounded scalar settings, never code, paths, network options, or callbacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from importlib.metadata import version

MAX_LINE = 262144
WARMUP_MARKER = "auto_mlx_runner_warmup_complete"
ITER_TIMINGS_MARKER = "auto_mlx_runner_iter_timings_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def forbidden(value):
        raise ValueError("floating/non-finite protocol values are not supported")
    require(len(raw) <= MAX_LINE, "protocol input exceeds limit")
    return json.loads(raw, object_pairs_hook=pairs, parse_float=forbidden, parse_constant=forbidden)


def emit(value):
    print(json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False), flush=True)


def config_value(value):
    require(type(value) is dict and set(value) == {"prefill_step_size"}, "expected only prefill_step_size")
    n = value["prefill_step_size"]
    require(type(n) is int and 128 <= n <= 4096, "prefill_step_size must be an integer in [128, 4096]")
    return n


def validate_request(value):
    require(type(value) is dict and set(value) == {"prompt", "max_tokens"}, "expected prompt and max_tokens")
    prompt, maximum = value["prompt"], value["max_tokens"]
    require(type(prompt) is str and 0 < len(prompt.encode("utf-8")) <= 65536, "prompt must contain 1..65536 UTF-8 bytes")
    require(type(maximum) is int and 1 <= maximum <= 4096, "max_tokens must be an integer in [1, 4096]")
    return prompt, maximum


def check_runtime(expected):
    actual = {"python": platform.python_version(), "system": platform.system(), "release": platform.release(), "machine": platform.machine()}
    for key, name in (("chip", "machdep.cpu.brand_string"), ("memory_bytes", "hw.memsize")):
        actual[key] = subprocess.check_output(("/usr/sbin/sysctl", "-n", name), text=True, timeout=2).strip() if platform.system() == "Darwin" else "unavailable"
    for package in ("mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "safetensors", "numpy"):
        actual[package] = version(package)
    require(actual == expected, "worker runtime does not match the pinned bundle")
    require(actual["system"] == "Darwin" and actual["machine"] == "arm64", "native adapter requires Apple Silicon")


def load_backend(root, bundle):
    check_runtime(bundle["runtime"])
    # Never use mlx_lm.load's Hub-or-path fallback. Both calls take an already
    # verified local snapshot; tokenizer network and remote code are disabled.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import mlx.core as mx
    # A reproducible eager baseline: do not require runtime shader compilation.
    # This does not alter the sandbox or fall back to a different device.
    mx.disable_compile()
    require(bundle["device"] in {"cpu", "gpu"}, "unsupported execution device")
    selected = mx.cpu if bundle["device"] == "cpu" else mx.gpu
    mx.set_default_device(selected)
    require(mx.default_device() == mx.Device(selected), "requested MLX device is unavailable")
    # MLX-LM creates its generation stream at import time: select the device first.
    from mlx_lm.utils import load_model, load_tokenizer
    model, config = load_model(Path(root), lazy=False, strict=True)
    require(config.get("model_type") == bundle["model_type"], "model type changed")
    tokenizer = load_tokenizer(Path(root), {"local_files_only": True, "trust_remote_code": False}, eos_token_ids=config.get("eos_token_id"))
    model.eval()
    return model, tokenizer


def generate_events(backend, bundle, request, prefill):
    prompt, maximum = validate_request(request)
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = backend
    add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    tokens = tokenizer.encode(prompt, add_special_tokens=add_special)
    require(bool(tokens), "prompt tokenization is empty")
    require(len(tokens) + maximum <= bundle["context_limit"], "prompt plus max_tokens exceeds the verified context limit")
    is_gpu = bundle["device"] == "gpu"
    if is_gpu:
        mx.reset_peak_memory()
    started = time.perf_counter_ns()
    first_token_ns = None
    ids, text = [], []
    detokenizer = tokenizer.detokenizer
    finish = "length"
    # The lower-level MLX-LM generator supports both explicit devices and
    # avoids the GPU-only wired-limit wrapper in stream_generate.
    stream = generate_step(mx.array(tokens), model, max_tokens=maximum, sampler=make_sampler(temp=0.0), prefill_step_size=prefill)
    try:
        for token, _ in stream:
            if first_token_ns is None:
                first_token_ns = time.perf_counter_ns() - started
            token = int(token)
            ids.append(token)
            if token in tokenizer.eos_token_ids:
                finish = "stop"
                detokenizer.finalize()
                piece = detokenizer.last_segment
                text.append(piece)
                yield {"event": "delta", "text": piece, "token": token}
                break
            detokenizer.add_token(token)
            if len(ids) == maximum:
                detokenizer.finalize()
            piece = detokenizer.last_segment
            text.append(piece)
            yield {"event": "delta", "text": piece, "token": token}
            if len(ids) == maximum:
                break
        elapsed = time.perf_counter_ns() - started
        require(first_token_ns is not None, "backend returned no tokens")
        if is_gpu:
            memory = int(mx.get_peak_memory())
        else:
            import resource
            memory = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        yield {"event": "done", "text": "".join(text), "token_ids": ids, "finish_reason": finish, "prompt_tokens": len(tokens), "completion_tokens": len(ids), "time_to_first_token_ns": first_token_ns, "generation_elapsed_ns": elapsed, "peak_memory_bytes": memory, "memory_metric": "metal_peak_allocated_bytes" if is_gpu else "process_peak_rss_bytes"}
    finally:
        stream.close()
        mx.synchronize()
        if is_gpu:
            mx.clear_cache()


def evaluate(root, bundle, prefill):
    backend = load_backend(root, bundle)
    count = int(os.environ.get("AUTO_MLX_K_REPETITIONS", "1"))
    require(1 <= count <= 10000, "invalid K repetitions")
    def iteration():
        values = []
        for fixture in bundle["fixtures"]:
            result = list(generate_events(backend, bundle, fixture, prefill))[-1]
            values.append({"text": result["text"], "token_ids": result["token_ids"], "finish_reason": result["finish_reason"]})
        return json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    expected = iteration()
    print(WARMUP_MARKER, file=sys.stderr, flush=True)
    timings = []
    for _ in range(count):
        started = time.perf_counter_ns()
        actual = iteration()
        timings.append(time.perf_counter_ns() - started)
        require(actual == expected, "greedy output was not repeatable within this runner")
    print(ITER_TIMINGS_MARKER + " " + json.dumps({"k": count, "iterations_ns": timings}, separators=(",", ":")), file=sys.stderr, flush=True)
    print(hashlib.sha256(expected).hexdigest(), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--force-baseline", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--python-site")
    args = parser.parse_args(argv)
    try:
        if args.python_site:
            require(Path(args.python_site).is_absolute() and Path(args.python_site).is_dir(), "invalid trusted Python site directory")
            sys.path.insert(0, args.python_site)
        bundle = loads(args.spec_json)
        root = os.environ.get("AUTO_MLX_ARTIFACT_ROOT")
        require(root is not None and Path(root).is_dir(), "verified artifact root is missing")
        if args.worker:
            backend = load_backend(root, bundle)
            emit({"event": "ready"})
            while True:
                raw = sys.stdin.buffer.readline(MAX_LINE + 1)
                if not raw:
                    break
                try:
                    require(raw.endswith(b"\n") and len(raw) <= MAX_LINE, "oversized worker request")
                    request = loads(raw)
                    require(type(request) is dict and set(request) == {"prompt", "max_tokens", "prefill_step_size"}, "invalid worker request fields")
                    prefill = config_value({"prefill_step_size": request.pop("prefill_step_size")})
                    for event in generate_events(backend, bundle, request, prefill):
                        emit(event)
                except (ValueError, TypeError, KeyError) as exc:
                    emit({"event": "error", "message": str(exc)})
            return 0
        path = os.environ.get("AUTO_MLX_CONFIG_PATH")
        require(bool(path), "candidate config is missing")
        with open(path, "rb") as handle:
            prefill = config_value(loads(handle.read(MAX_LINE + 1)))
        evaluate(root, bundle, 2048 if args.force_baseline else prefill)
        return 0
    except Exception as exc:
        if args.worker:
            emit({"event": "error", "message": f"native worker failed: {type(exc).__name__}: {exc}"})
        else:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"native MLX-LM runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
