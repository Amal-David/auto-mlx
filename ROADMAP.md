# Roadmap

The roadmap is staged by evidence, not by how exciting a technique sounds.

## G0 — implemented

- Strict canonical JSON and stable failure codes.
- Immutable workload, knob, provider, artifact, policy, runtime, and candidate contracts.
- Evaluator-derived candidate IDs.
- Safe local artifact identity checks.
- Dry-run-first validation and ID inspection with no MLX import.

## G1 — evaluator library and local CLI evaluation implemented

The evaluator-owned execution policy, trusted-runner boundary, exact-output oracle, and `Evaluator` library are implemented. `auto-mlx evaluate` wires them into the CLI on the local sandbox tier (`auto_mlx.sandbox`, macOS `sandbox-exec`): it runs a real sandboxed baseline/candidate comparison, builds a receipt, and attempts local supervisor attestation. Without the local sandbox execution primitives, `evaluate` fails closed with an `unavailable` diagnostic and exit code 4 — a missing Apple Silicon/MLX/sandbox environment is always a stated unavailable result, never a speedup result. Real, hardened production-grade isolation beyond this local, single-operator tier remains later work.

## G2 — receipt, promotion, and dispatch libraries implemented; local activation wired, production gate remains

Immutable content-addressed receipts, independent aggregate recomputation, promotion decisions, rollback, and exact identity dispatch with native fallback exist as library controls. `auto-mlx promote`, `dispatch`, `rollback`, and `keys ensure` wire them into the CLI: `promote` independently re-verifies and re-attests a stored receipt through the local supervisor (`auto_mlx.supervisor`) and persists an activation or native-fallback decision; `dispatch` resolves, and with `--execute` actually runs, the currently active candidate or the native fallback. Promotion rejects tampering, incomplete samples, identity drift, runtime failures, and correctness regressions, and only activates on a positive, well-formed, independently recomputed gain. This activation gate is local-supervisor-attested, not a hardened, multi-party, or fleet-wide production activation gate — that remains later work.

## G2.5 — statistical decisions and knob-grid search, implemented locally

Decisions are no longer the sign of a single difference. Each sample is the minimum of K in-process
iterations timed inside one sandboxed runner launch; self-reported iteration timings are cross-checked
against the evaluator's own wall-clock span and rejected as forged when implausible, so no self-reported
number is ever evidentiary. Verdicts come from a bootstrap confidence interval over paired blocks,
extended sequentially until the interval resolves or a cap is reached, against a min-effect threshold
that an A/A calibration mode measures empirically instead of asserting. `inconclusive` is a first-class
verdict, distinct from a measured regression, and promotion refuses it.

`auto-mlx tune` races a declarative provider's knob grid against a permanent, unremovable baseline floor,
eliminating candidates only on statistical futility or a decisive regression, honoring explicit budgets,
and storing a content-addressed tuning summary plus a full attested receipt for every measured candidate.
`auto-mlx history` reads those summaries back per workload identity, and a tune reuses prior results only
when both workload hash and runtime identity match.

Measured on this repository's toy workload, the search correctly finds no winner: that example's `tile`
knob is inert by construction and `eager` versus `compiled` is indistinguishable at the host's measured
noise floor. Learned cost models, persistent warm measurement workers, and realistic LLM prefill/decode
workloads (where genuine wins are expected to live) are not implemented and remain later work.

## G3 — compiler search and custom Metal, deferred

Compiler search and custom Metal kernels remain research work. They require a measured baseline, a native MLX fallback, correctness/parity tests, workload and hardware metadata, a rollback condition, and a reproducible local receipt before they can be called supported. No G3 speedup has been measured.
