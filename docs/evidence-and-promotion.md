# Evidence and promotion

Evaluation and promotion are separate authorities.

1. The `Evaluator` library retains declarative-candidate observations. It only ever launches a candidate through an evaluator-owned, explicitly constructed isolation provider/authority pair; on a host without the local sandbox execution primitives (`auto_mlx.executor.local_sandbox_primitives_available`), or without a caller-supplied provider/authority, it records `SANDBOX_UNAVAILABLE`. `auto_mlx.sandbox` implements the local tier: macOS `sandbox-exec` (Seatbelt), independently probed by its own authority -- see `docs/threat-model.md` for exactly what that tier does and does not guarantee (a single-user, developer-grade guard, not a hardened multi-tenant boundary).
2. The checked-in receipt store uses exclusive immutable creation and binds the receipt to canonical identities.
3. An independent promotion verifier must recompute aggregates from the retained samples, recheck correctness and identity, and reject drift or incomplete evidence. `auto_mlx.supervisor.attest_receipt` is the sole code path allowed to mint the HMAC attestation that later gates activation; it independently reruns every one of these checks itself before signing.
4. The promotion and dispatch libraries implement the decision and exact-match/native-fallback boundaries. Only a promotion result may make a candidate eligible for exact-match dispatch. Any identity mismatch, runtime failure, missing evidence, or correctness regression must use the native fallback. Activation is gated on local supervisor attestation plus a positive, well-formed, independently recomputed gain; a hardened, multi-party, or fleet-wide production activation gate remains later work.

`auto-mlx evaluate`, `promote`, `dispatch`, and `rollback` wire the evaluator, receipt, local supervisor attestation, promotion, and dispatch libraries into the CLI (see `docs/cli.md`). `evaluate` and `dispatch --execute` require the local sandbox execution primitives; without them they return a JSON `unavailable` diagnostic with exit code 4, never a successful no-op. `promote`, plain `dispatch`, `rollback`, and `keys ensure` never need the sandbox and always run.

## Promotion must not mean publication

Even a future local promotion is not permission to publish a general performance claim. A public claim needs a reproducible workload, pinned software and hardware, baseline definition, complete paired measurements, quality gate, and a link to the immutable evidence. Local observations stay local until those requirements are met.

## Rollback

Promotion must be reversible. A later runtime or artifact identity mismatch, failed correctness check, receipt tamper, or repeated measurement regression resolves to the native path and leaves the prior receipt intact.
