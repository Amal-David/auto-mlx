# Evidence and promotion

Evaluation and promotion are separate authorities.

1. The `Evaluator` library may run a declarative candidate against a frozen workload and record all baseline and candidate samples; production evaluation may fail closed until real isolation and supervisor proof are available.
2. The checked-in receipt store uses exclusive immutable creation and binds the receipt to canonical identities.
3. An independent promotion verifier must recompute aggregates from the retained samples, recheck correctness and identity, and reject drift or incomplete evidence.
4. The promotion and dispatch libraries implement the decision and exact-match/native-fallback boundaries. Only a promotion result may make a candidate eligible for exact-match dispatch. Any identity mismatch, runtime failure, missing evidence, or correctness regression must use the native fallback, and production activation remains gated on later evidence and activation proof.

The current checkout implements evaluator, receipt, promotion, and dispatch libraries. The `evaluate`, `promote`, and `dispatch` CLI commands return a JSON `unavailable` diagnostic with a nonzero exit because CLI orchestration is deferred; they are not successful no-op stubs.

## Promotion must not mean publication

Even a future local promotion is not permission to publish a general performance claim. A public claim needs a reproducible workload, pinned software and hardware, baseline definition, complete paired measurements, quality gate, and a link to the immutable evidence. Local observations stay local until those requirements are met.

## Rollback

Promotion must be reversible. A later runtime or artifact identity mismatch, failed correctness check, receipt tamper, or repeated measurement regression resolves to the native path and leaves the prior receipt intact.
