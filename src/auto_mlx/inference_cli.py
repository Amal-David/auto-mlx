"""CLI glue for native bundles, generation and localhost serving (stdlib only)."""
from __future__ import annotations

import signal
from pathlib import Path
import time

from .canonical import canonical_bytes, canonical_json
from .contracts import CandidateProposal, EvaluationPolicy, RuntimeIdentity
from .dispatch import dispatch, CANDIDATE_MODE
from .errors import KeyMaterialError
from . import keys, store_config
from .model_bundle import ModelBundle, prepare_bundle, require, BASELINE_PREFILL
from .providers import DeclarativeProvider


def add_arguments(subparsers, parser_class):
    bundle = subparsers.add_parser("bundle", help="prepare, inspect, verify or export a native model bundle", allow_abbrev=False)
    children = bundle.add_subparsers(dest="bundle_command", required=True, parser_class=parser_class)
    prepare = children.add_parser("prepare", allow_abbrev=False)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--source-id", required=True)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--license", required=True)
    prepare.add_argument("--fixtures", required=True, help="JSON array of raw prompt / max_tokens fixtures")
    prepare.add_argument("--context-limit", type=int)
    prepare.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="explicit device; never silently falls back")
    prepare.add_argument("--output", required=True)
    for command in ("inspect", "verify", "workload"):
        child = children.add_parser(command, allow_abbrev=False)
        child.add_argument("--bundle", required=True)
        if command == "verify":
            child.add_argument("--model-root", required=True)
            child.add_argument("--execute", action="store_true")
            child.add_argument("--timeout-seconds", type=int, default=120)
            child.add_argument("--output")
        if command == "workload":
            child.add_argument("--output", required=True)
            child.add_argument("--provider-output")
            child.add_argument("--candidate-output")
    for command in ("generate", "serve"):
        child = subparsers.add_parser(command, allow_abbrev=False, help="native local inference; dry-run unless --execute")
        child.add_argument("--bundle", required=True)
        child.add_argument("--model-root", required=True)
        child.add_argument("--execute", action="store_true")
        child.add_argument("--timeout-seconds", type=int, default=120)
        child.add_argument("--candidate", help="optional candidate whose activation is independently verified")
        child.add_argument("--policy", help="policy used to evaluate the candidate")
        child.add_argument("--store")
        child.add_argument("--key-dir")
        if command == "generate":
            child.add_argument("--prompt", required=True)
            child.add_argument("--max-tokens", type=int, default=128)
            child.add_argument("--stream", action="store_true", help="emit newline-delimited JSON events")
        else:
            child.add_argument("--port", type=int, default=8000, help="bind only 127.0.0.1 at this port")


def _bundle(args):
    from .cli import _read_json
    return ModelBundle.from_dict(_read_json(args.bundle))


def run_bundle(args):
    from .cli import _read_json, _create_only, _require_local_sandbox
    if args.bundle_command == "prepare":
        bundle = prepare_bundle(Path(args.model), source={"id": args.source_id, "revision": args.revision, "license": args.license}, fixtures=_read_json(args.fixtures), context_limit=args.context_limit, device=args.device)
        _create_only(args.output, bundle.payload + b"\n")
        return {"ok": True, "bundle_id": bundle.bundle_id, "output": args.output, "verification": "artifact-inventory-only", "source_parity": "not-established"}
    bundle = _bundle(args)
    if args.bundle_command == "inspect":
        return {"ok": True, "bundle_id": bundle.bundle_id, "bundle": bundle.to_dict(), "verification": "document-only", "source_parity": "not-established"}
    if args.bundle_command == "workload":
        workload = bundle.workload()
        provider = DeclarativeProvider("mlx-lm-prefill-v1", ({"prefill_step_size": 128}, {"prefill_step_size": 512}, {"prefill_step_size": BASELINE_PREFILL}))
        candidate = CandidateProposal(provider.provider_id, workload, {"prefill_step_size": 512})
        outputs = [(args.output, workload.to_dict()), (args.provider_output, provider.to_dict()), (args.candidate_output, candidate.to_dict())]
        paths = [p for p, _ in outputs if p]
        require(len(paths) == len(set(paths)), "output paths must be distinct")
        require(all(not Path(p).exists() and not Path(p).is_symlink() for p in paths), "output already exists")
        for path, value in outputs:
            if path:
                _create_only(path, canonical_bytes(value) + b"\n")
        return {"ok": True, "bundle_id": bundle.bundle_id, "workload_hash": workload.workload_hash, "outputs": paths}
    bundle.verify(Path(args.model_root), check_runtime=args.execute)
    result = {"ok": True, "bundle_id": bundle.bundle_id, "artifacts_verified": True, "source_parity": "not-established", "native_repeatability": "not-run"}
    if args.execute:
        _require_local_sandbox(stage="native-verification")
        from .inference import InferenceSession
        observations = []
        with InferenceSession(bundle, Path(args.model_root), timeout_seconds=args.timeout_seconds) as session:
            for fixture in bundle.to_dict()["fixtures"]:
                baseline = session.generate(fixture)
                repeated = session.generate(fixture)
                fields = ("text", "token_ids", "finish_reason")
                require(all(baseline[k] == repeated[k] for k in fields), "native greedy repeatability failed")
                observations.append({"fixture": fixture, "baseline": baseline, "repeat": repeated})
        result.update({"native_repeatability": "passed", "observations": observations, "scope": "native-repeatability-not-source-parity", "created_at_ns": time.time_ns()})
    if args.output:
        _create_only(args.output, canonical_bytes(result) + b"\n")
    return result


def profile_resolver(bundle, root, args):
    from .cli import _read_json
    candidate = None
    if args.candidate:
        require(bool(args.policy), "a candidate requires its evaluation --policy")
        candidate = CandidateProposal.from_dict(_read_json(args.candidate), bundle.workload())
        policy = EvaluationPolicy.from_dict(_read_json(args.policy))
    def resolve(request):
        native = {"mode": "native_fallback", "prefill_step_size": BASELINE_PREFILL, "reason": "no_candidate_requested"}
        if candidate is None:
            return native
        # A receipt for fixed prompts does not prove an optimization on a new
        # prompt/context. Never apply its configuration outside those fixtures.
        if request not in bundle.to_dict()["fixtures"]:
            return {**native, "reason": "request_outside_verified_workload"}
        store = store_config.open_store(args.store, key_dir=args.key_dir)
        try:
            key = keys.load_attestation_key(key_dir=args.key_dir)
        except KeyMaterialError:
            key = None
        result = dispatch(store, bundle.workload(), candidate, policy, RuntimeIdentity.current(), artifact_root=str(root), attestation_key=key, now_ns=time.time_ns())
        return {"mode": result.mode, "prefill_step_size": candidate.config["prefill_step_size"] if result.mode == CANDIDATE_MODE else BASELINE_PREFILL, "reason": result.reason, "receipt_id": result.receipt_id}
    return resolve


def run_inference(args):
    from .cli import _write_stdout, _require_local_sandbox
    from .runners.mlx_lm_runner import validate_request
    from .model_bundle import integer
    bundle = _bundle(args)
    root = Path(args.model_root)
    bundle.verify(root, check_runtime=args.execute)
    integer(args.timeout_seconds, 1, 600, "timeout_seconds")
    if args.command == "generate":
        request = {"prompt": args.prompt, "max_tokens": args.max_tokens}
        try:
            validate_request(request)
        except ValueError as exc:
            require(False, str(exc))
    else:
        integer(args.port, 0, 65535, "port")
    resolve = profile_resolver(bundle, root, args)
    if not args.execute:
        return {"ok": True, "command": args.command, "dry_run": True, "bundle_id": bundle.bundle_id, "host": "127.0.0.1" if args.command == "serve" else None, "source_parity": "not-established"}
    _require_local_sandbox(stage="native-inference")
    from .inference import InferenceSession
    with InferenceSession(bundle, root, timeout_seconds=args.timeout_seconds) as session:
        if args.command == "generate":
            profile = resolve(request)
            if not args.stream:
                return {"ok": True, "command": "generate", "profile": profile, "result": session.generate(request, prefill_step_size=profile["prefill_step_size"])}
            stream = session.stream(request, prefill_step_size=profile["prefill_step_size"])
            try:
                for event in stream:
                    _write_stdout(canonical_json(event) + "\n")
            finally:
                stream.close()
            return {"ok": True, "command": "generate", "complete": True, "profile": profile}
        from .inference_server import LocalInferenceServer
        session.warmup()
        server = LocalInferenceServer(args.port, session, resolve)
        def terminate(*_):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            _write_stdout(canonical_json({"event": "server_ready", "host": "127.0.0.1", "port": server.server_port, "bundle_id": bundle.bundle_id}) + "\n")
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            signal.signal(signal.SIGTERM, previous)
        return {"ok": True, "command": "serve", "stopped": True}
