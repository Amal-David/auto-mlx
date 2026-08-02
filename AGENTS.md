# Agent guidance

This directory is a standalone, standard-library Python 3.11+ project. Work from the existing contract classes and JSON schemas; do not widen their public invariants casually.

- Keep the G0 CLI offline, dry-run-first, and importable without MLX.
- Keep JSON output on stdout and diagnostics on stderr with stable nonzero exits.
- Treat candidate inputs as declarative data only. Never add shell, remote code, network, secret, or arbitrary callback surfaces.
- Preserve canonical identity, evaluator-derived IDs, safe artifact checks, and exact-field rejection.
- Evaluator, receipt, promotion, and dispatch libraries exist, but production evaluation/activation must fail closed until their isolation, supervisor, evidence, and activation gates exist; CLI orchestration, compiler search, and custom Metal work remain deferred.
- Every optimization proposal needs a correctness check, benchmark metadata, and a rollback condition.
- Run the complete offline unittest suite before handing work back.
