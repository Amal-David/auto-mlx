# Roadmap

The roadmap is staged by evidence, not by how exciting a technique sounds.

## G0 — implemented

- Strict canonical JSON and stable failure codes.
- Immutable workload, knob, provider, artifact, policy, runtime, and candidate contracts.
- Evaluator-derived candidate IDs.
- Safe local artifact identity checks.
- Dry-run-first validation and ID inspection with no MLX import.

## G1 — evaluator library implemented; CLI orchestration deferred

The evaluator-owned execution policy, trusted-runner boundary, exact-output oracle, and `Evaluator` library are implemented. Production evaluation may fail closed until real isolation and supervisor proof are available. CLI orchestration must use a pinned source oracle, declarative candidates, bounded subprocess resources, sanitized inputs, complete paired samples, exact-output correctness, and explicit capability checks. A missing Apple Silicon/MLX environment must be a stated unavailable result, never a speedup result.

## G2 — receipt, promotion, and dispatch libraries implemented; activation gate remains

Immutable content-addressed receipts, independent aggregate recomputation, promotion decisions, rollback, and exact identity dispatch with native fallback exist as library controls. Production activation remains gated on later evidence and activation proof, and CLI orchestration remains deferred. Promotion must reject tampering, incomplete samples, identity drift, runtime failures, and correctness regressions.

## G3 — compiler search and custom Metal, deferred

Compiler search and custom Metal kernels remain research work. They require a measured baseline, a native MLX fallback, correctness/parity tests, workload and hardware metadata, a rollback condition, and a reproducible local receipt before they can be called supported. No G3 speedup has been measured.
