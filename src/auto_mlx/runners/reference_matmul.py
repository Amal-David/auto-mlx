"""Standalone reference MLX runner for the G0/G1 evaluator's toy-matmul workload.

This module is executed as its own subprocess (``[sys.executable,
<this file>]``), never imported by the evaluator core.  Two invariants keep
that safe:

1. ``mlx`` is imported strictly inside functions, never at module scope, so
   merely importing this module -- e.g. ``import
   auto_mlx.runners.reference_matmul`` -- never puts ``mlx`` into
   ``sys.modules``.  ``auto_mlx`` (the evaluator/CLI package) must stay
   importable on hosts without MLX installed at all.
2. This module never imports ``auto_mlx`` itself.  It reads its one input
   (the knob config) from the ``AUTO_MLX_CONFIG_PATH`` environment variable
   that ``execute_plan`` always sets, and writes its one output (a result
   digest) to stdout -- the same evaluator-owned contract every other
   trusted runner in this repository follows.

Workload contract
------------------

The shape and dtype below are hardcoded to stay consistent with
``examples/workload.json``'s ``toy-matmul`` workload
(``parameters.dtype == "float32"``, ``parameters.shape == [1, 3072, 3072]``;
the leading ``1`` is a reserved, currently-unused batch dimension).  Nothing
at runtime reads ``workload.json`` -- the two are kept in sync by hand and
checked by ``tests/test_runner_reference.py`` and by the example set's own
CLI validation.  Only two knobs are accepted, matching the workload's
declared knobs exactly:

- ``mode``: ``"eager"`` or ``"compiled"``.
- ``tile``: an integer in ``[16, 32]``.

Unknown keys, wrong types, or out-of-range values are a fail-closed,
nonzero-exit configuration error (diagnostics on stderr, nothing on
stdout).

Why ``tile`` is validated but not applied
------------------------------------------

Tiling a matmul changes float32 accumulation order (partial sums are
combined in a different sequence), and this runner's whole job is to stay
byte-identical against ``ExactOutputOracle`` across every accepted
config -- including across ``mode`` -- on this machine (verified
empirically: 10 back-to-back runs each of eager and compiled produced the
identical digest below, and eager and compiled matched each other exactly).
Shipping a real tiled variant would either break that exact-match contract
or require a per-tile oracle, which this evaluator generation does not
support.  So ``tile`` is accepted and bounds-checked -- a candidate that
sends a bad ``tile`` still fails closed -- but intentionally left
uncomputed for now.  A future wave that wants tiling to actually change the
result will need either an approximate oracle or per-tile expected outputs.

Runner contract: one uncounted warmup before the measured run
----------------------------------------------------------------

``mx.compile`` has an OS-level (Metal shader cache) cold/warm state that
persists across *processes*, not just within one: the first-ever compiled
launch on a machine measured ~1151ms versus ~140ms once warm, while eager
mode carries no such cost.  Because each evaluator sample is a fresh
subprocess and the evaluator forces the baseline arm to always run eager,
that cold tax previously landed asymmetrically on the compiled candidate
arm alone -- a structural confound, not a real baseline-vs-candidate
difference (see ``docs/measurement.md``).

This runner now performs exactly one full, uncounted execution of the same
computation (build inputs, run ``fn``, ``mx.eval``) immediately before the
measured run, discarding its result.  The digest the oracle checks always
comes from the *last* measured run (see "K in-runner timed iterations"
below); the warmup exists purely to pay compile/shader-cache/page-fault
costs up front so the timed runs are warm regardless of process cold-start
state.  After the warmup completes, this runner prints a fixed marker line
to stderr (never stdout -- stdout stays the oracle-sacred single digest
line) so the evaluator can record an honest, non-evidentiary ``warm_state``
note in the receipt.  The marker string is intentionally duplicated in
``auto_mlx.receipts._WARMUP_MARKER`` rather than imported, per this
module's own no-``auto_mlx``-imports invariant above; keep the two literals
identical if either changes.

Runner contract: K in-runner timed iterations (Wave B)
----------------------------------------------------------------

After the uncounted warmup above, this runner performs ``K`` additional
eval-fenced executions of the same computation, timing each one with
``time.perf_counter_ns()``.  ``K`` comes from the ``AUTO_MLX_K_REPETITIONS``
environment variable the evaluator sets (see
``auto_mlx.evaluator._execution_policy_from_contract``); it defaults to
``DEFAULT_K_REPETITIONS`` below when the variable is absent (manual/ad-hoc
invocations).  The oracle-sacred stdout digest still comes from exactly one
line -- the digest of the *last* of the K measured iterations (the
workload is deterministic, so every iteration's digest is identical; see
"Why ``tile`` is validated but not applied" above).

The per-iteration timings are reported on a second, non-oracle stderr line
prefixed with ``ITER_TIMINGS_MARKER`` and followed by a single-line JSON
object ``{"k": K, "iterations_ns": [...]}``.  This channel is
self-reported and is not evidentiary on its own: the evaluator's parent
process wall-clock span (``ExecutionRecord.runner_elapsed_ns``) remains the
sole evidentiary timing anchor.  ``auto_mlx.statistics`` cross-checks the
reported sum against that parent-observed span before trusting it at all
(see ``auto_mlx.statistics.compute_sample_timing`` and its
``forged_timing`` rejection); an untrusted report degrades the sample to
K=1 semantics (the parent span alone), never silently accepted.  The marker
string is intentionally duplicated in
``auto_mlx.statistics._ITER_TIMINGS_MARKER`` rather than imported, for the
same reason ``WARMUP_MARKER`` is duplicated above.

Device: CPU, deliberately
--------------------------

MLX defaults to the GPU (``Device(gpu, 0)``) on Apple Silicon.  This runner
explicitly calls ``mx.set_default_device(mx.cpu)`` instead.  That was a
choice, not a fallback forced by a failure: probed directly (outside this
runner, see the reference-matmul design notes / task return), the exact
``LocalSandboxProvider`` Seatbelt profile in ``auto_mlx.sandbox`` --
unmodified, no extra allowances -- let a GPU matmul run and produced a
digest that was bit-identical across repeated sandboxed runs and identical
to the CPU digest for this workload. GPU is not blocked here. CPU was still
chosen as the shipped default for this wave because it is the more
conservative, universally-available baseline (works identically on any Mac,
with or without a capable GPU/Metal driver, and its determinism story does
not depend on Metal's own kernel scheduling being repeatable across macOS
versions or hardware). Switching the default to GPU is a reasonable future
change once that determinism claim has more than one machine's evidence
behind it -- tracked as follow-up, not done here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time


# Hardcoded to match examples/workload.json's toy-matmul parameters:
# dtype "float32", shape [1, 3072, 3072] (leading 1 is a reserved,
# currently-unused batch dimension -- this runner only ever builds a single
# (MATRIX_SIZE, MATRIX_SIZE) pair).
MATRIX_SIZE = 3072
ALLOWED_KEYS = frozenset({"mode", "tile"})
ALLOWED_MODES = ("eager", "compiled")
TILE_MINIMUM = 16
TILE_MAXIMUM = 32
# Must stay byte-identical to auto_mlx.receipts._WARMUP_MARKER (this module
# cannot import auto_mlx -- see the module docstring).
WARMUP_MARKER = "auto_mlx_runner_warmup_complete"
# Must stay byte-identical to auto_mlx.statistics._ITER_TIMINGS_MARKER (this
# module cannot import auto_mlx -- see the module docstring).
ITER_TIMINGS_MARKER = "auto_mlx_runner_iter_timings_v1"
# Used only when AUTO_MLX_K_REPETITIONS is absent (manual/ad-hoc
# invocations); the evaluator always sets the environment variable
# explicitly from EvaluationPolicy.k_repetitions (default also 50).
DEFAULT_K_REPETITIONS = 50
MAX_K_REPETITIONS = 10_000
_K_REPETITIONS_ENV = "AUTO_MLX_K_REPETITIONS"


class ConfigError(ValueError):
    """A knob config that fails this runner's fail-closed contract."""


def parse_config(raw: bytes) -> tuple[str, int]:
    """Parse and strictly validate the runner's one JSON input.

    Returns ``(mode, tile)``.  Raises :class:`ConfigError` -- never a bare
    exception -- for anything outside the exact ``{"mode": ..., "tile":
    ...}`` contract: unknown keys, missing keys, wrong types, or an
    out-of-range ``tile``.
    """

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config is not valid JSON: {exc}") from exc
    if type(config) is not dict:
        raise ConfigError("config must be a JSON object")
    unknown = set(config) - ALLOWED_KEYS
    if unknown:
        raise ConfigError(f"unknown knob(s): {sorted(unknown)}")
    missing = ALLOWED_KEYS - set(config)
    if missing:
        raise ConfigError(f"missing knob(s): {sorted(missing)}")
    mode = config["mode"]
    if type(mode) is not str or mode not in ALLOWED_MODES:
        raise ConfigError(f"mode must be one of {ALLOWED_MODES}, got {mode!r}")
    tile = config["tile"]
    if type(tile) is not int:
        raise ConfigError(f"tile must be an integer, got {tile!r}")
    if not (TILE_MINIMUM <= tile <= TILE_MAXIMUM):
        raise ConfigError(f"tile must be within [{TILE_MINIMUM}, {TILE_MAXIMUM}], got {tile}")
    return mode, tile


def _build_inputs(mx: object, n: int):
    """Deterministic, RNG-free float32 operands derived from ``mx.arange``.

    No randomness anywhere: the same ``n`` always produces the same two
    matrices, on any host, any run.  Values are bounded to roughly
    ``[-0.5, 0.5)`` so the matmul result stays well inside float32 range
    for the matrix sizes this runner uses.
    """

    idx = mx.arange(n * n, dtype=mx.float32)
    a = (mx.remainder(idx * 7.0 + 3.0, 17.0) / 17.0 - 0.5).reshape(n, n)
    b = (mx.remainder(idx * 5.0 + 11.0, 13.0) / 13.0 - 0.5).reshape(n, n)
    mx.eval(a, b)
    return a, b


def _matmul(a: object, b: object) -> object:
    import mlx.core as mx

    return mx.matmul(a, b)


def _digest_result(result: object) -> str:
    """Canonical digest: an explicit dtype/shape header plus the raw buffer.

    The header guards against two differently-shaped or differently-typed
    results that happen to share raw bytes ever being mistaken for the same
    digest.
    """

    digest = hashlib.sha256()
    digest.update(str(result.dtype).encode("ascii"))
    digest.update(repr(tuple(result.shape)).encode("ascii"))
    digest.update(bytes(memoryview(result)))
    return digest.hexdigest()


def _parse_k_repetitions(raw: str | None) -> int:
    """Parse ``AUTO_MLX_K_REPETITIONS``; fail closed on anything malformed.

    Absent -> :data:`DEFAULT_K_REPETITIONS`.  Present but not a positive
    integer within ``[1, MAX_K_REPETITIONS]`` -> :class:`ConfigError`, same
    fail-closed contract as every other input this runner parses.
    """

    if raw is None:
        return DEFAULT_K_REPETITIONS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{_K_REPETITIONS_ENV} must be an integer, got {raw!r}") from exc
    if not (1 <= value <= MAX_K_REPETITIONS):
        raise ConfigError(f"{_K_REPETITIONS_ENV} must be within [1, {MAX_K_REPETITIONS}], got {value}")
    return value


def run(mode: str, tile: int, *, k_repetitions: int) -> str:
    """Execute the toy-matmul workload and return its canonical result digest.

    ``tile`` is accepted and was already bounds-checked by
    :func:`parse_config`; see the module docstring for why it is not
    applied to the computation.  ``k_repetitions`` is the count of
    additional, eval-fenced timed iterations run after the one uncounted
    warmup (see the module docstring's "K in-runner timed iterations"
    section); the returned digest always comes from the last of those K
    iterations.
    """

    del tile  # validated, intentionally inert -- see module docstring
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    a, b = _build_inputs(mx, MATRIX_SIZE)
    fn = mx.compile(_matmul) if mode == "compiled" else _matmul

    # One uncounted warmup: see the module docstring's "Runner contract"
    # section.  Same `fn`, same inputs, discarded result -- pays compile/
    # shader-cache/page-fault cost before the measured, digested runs.
    warmup_result = fn(a, b)
    mx.eval(warmup_result)
    print(WARMUP_MARKER, file=sys.stderr, flush=True)

    # K eval-fenced timed iterations.  Each iteration is timed individually
    # with a monotonic, high-resolution clock; the parent evaluator process
    # never trusts this self-reported array on its own (see the module
    # docstring and auto_mlx.statistics.compute_sample_timing) -- it is
    # cross-checked against the parent-observed subprocess wall-clock span
    # before being used for anything.
    iterations_ns: list[int] = []
    result = None
    for _ in range(k_repetitions):
        started = time.perf_counter_ns()
        result = fn(a, b)
        mx.eval(result)
        iterations_ns.append(time.perf_counter_ns() - started)
    timings_payload = json.dumps({"k": k_repetitions, "iterations_ns": iterations_ns}, separators=(",", ":"))
    print(f"{ITER_TIMINGS_MARKER} {timings_payload}", file=sys.stderr, flush=True)

    return _digest_result(result)


def _parse_argv(argv: list[str]) -> str | None:
    """Parse this runner's only accepted flag: an evaluator-pinned mode override.

    ``--force-mode=<mode>`` exists so a ``TrustedRunner`` registration can
    bind, immutably, a "baseline" identity that always runs in ``eager``
    mode regardless of what a candidate's shared config says -- see
    ``auto_mlx.runners.reference.register_reference_matmul_runners``.  Since
    the flag lives in the runner's ``argv`` (part of its verified digest
    binding, not part of the untrusted candidate config), it cannot be
    influenced by a candidate proposal.
    """

    force_mode: str | None = None
    for argument in argv:
        if argument.startswith("--force-mode="):
            force_mode = argument.split("=", 1)[1]
        else:
            raise ConfigError(f"unrecognized argument: {argument!r}")
    if force_mode is not None and force_mode not in ALLOWED_MODES:
        raise ConfigError(f"--force-mode must be one of {ALLOWED_MODES}, got {force_mode!r}")
    return force_mode


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        force_mode = _parse_argv(arguments)
        config_path = os.environ.get("AUTO_MLX_CONFIG_PATH")
        if not config_path:
            raise ConfigError("AUTO_MLX_CONFIG_PATH is not set")
        with open(config_path, "rb") as handle:
            raw = handle.read()
        mode, tile = parse_config(raw)
        if force_mode is not None:
            mode = force_mode
        k_repetitions = _parse_k_repetitions(os.environ.get(_K_REPETITIONS_ENV))
        digest = run(mode, tile, k_repetitions=k_repetitions)
    except ConfigError as exc:
        print(f"invalid reference_matmul runner input: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"reference_matmul runner could not read its config: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # fail closed on any unexpected runtime error
        print(f"reference_matmul runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
